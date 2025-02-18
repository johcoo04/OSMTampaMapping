import logging
import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
from process_json import load_centroids, load_population_data
import process_geojson as pgj
import process_geoDataFrames as pgdf
from visualize import visualize, draw_graph  # Import draw_graph
from create_graph import create_graph
from shapely import wkt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration file
with open('config.json') as config_file:
    config = json.load(config_file)

def long_processing():
    ''' Function to load and process original GeoJSON data '''

    # Create ZIP code centroids and save to JSON
    logger.info("Creating ZIP code centroids and saving to JSON")
    centroids_gdf = pgj.create_zip_code_centroids(config['zip_geojson_path'], config['centroids_path'])

    # Load evacuation routes GeoJSON
    logger.info("Loading evacuation routes GeoJSON")
    routes_gdf = pgj.load_routes(config['routes_geojson_path'])

    # Load ramps GeoJSON
    logger.info("Loading ramps GeoJSON")
    ramps_gdf = pgj.load_ramps(config['ramps_geojson_path'])

    # Filter ramps near highways
    logger.info("Filtering ramps within 500 meters of the highways")
    filtered_ramps_gdf = pgdf.filter_ramps_near_highways(ramps_gdf, routes_gdf, distance=500)
    filtered_ramps_gdf.to_file(config['filtered_ramps_path'], driver="GeoJSON")

    # Merge close nodes within 500 meters
    logger.info("Merging close nodes within 500 meters")
    merged_ramp_nodes_gdf = pgdf.merge_close_nodes(filtered_ramps_gdf, distance=500)
    merged_ramp_nodes_gdf.to_file(config['merged_ramp_nodes_path'], driver="GeoJSON")

    # Load intersections GeoJSON
    logger.info("Loading intersections GeoJSON")
    intersections_gdf = pgj.load_intersections(config['intersections_geojson_path'])

    # Filter intersections near evacuation routes
    logger.info("Filtering intersections within 50 meters of evacuation routes")
    filtered_intersections_gdf = pgdf.filter_intersections_near_route(intersections_gdf, routes_gdf, distance=50)
    filtered_intersections_gdf.to_file(config['filtered_intersections_path'], driver="GeoJSON")

    # Merge close intersections within 500 meters
    logger.info("Merging close intersections within 500 meters")
    merged_intersection_nodes_gdf = pgdf.merge_close_nodes(filtered_intersections_gdf, distance=500)
    merged_intersection_nodes_gdf.to_file(config['merged_intersection_nodes_path'], driver="GeoJSON")

    # Trim intersections to include only every 5th node
    logger.info("Trimming intersections to include every 5th node")
    trimmed_intersections_gdf = pgdf.trim_intersections(merged_intersection_nodes_gdf, step=5)
    trimmed_intersections_gdf.to_file(config['trimmed_intersections_path'], driver="GeoJSON")

    # Calculate minimum distances from centroids to evacuation routes
    logger.info("Calculating minimum distances from centroids to closest point on evacuation routes")
    route_min_distances, route_lines = pgdf.calculate_min_distances_to_routes(centroids_gdf, routes_gdf)

    # Calculate minimum distances from centroids to ramps
    logger.info("Calculating minimum distances from centroids to ramps")
    ramp_min_distances, ramp_lines = pgdf.calculate_min_distances_to_ramps(centroids_gdf, merged_ramp_nodes_gdf, num_paths=1)

    # Combine the results into a single DataFrame
    min_distances_df = pd.DataFrame({
        'ZipCode': centroids_gdf['ZipCode'],
        'route_min_distance': route_min_distances,
        'ramp_min_distance': ramp_min_distances,
        'route_lines': route_lines,
        'ramp_lines': ramp_lines
    })

    # Save the results to a CSV file
    min_distances_df.to_csv(config['min_distances_path'], index=False)
    logger.info(f"Minimum distances saved to {config['min_distances_path']}")

    # Create GeoDataFrames for the lines
    route_lines_gdf = gpd.GeoDataFrame(geometry=route_lines, crs="EPSG:3857")
    ramp_lines_gdf = gpd.GeoDataFrame(geometry=ramp_lines, crs="EPSG:3857")

    # Create nodes for intersections and route points
    # logger.info("Creating nodes along routes")
    # route_nodes_gdf = pgdf.create_nodes_along_routes(routes_gdf, interval=500)

    # Create a graph from the GeoDataFrames
    logger.info("Creating a graph from the GeoDataFrames")
    G = create_graph(centroids_gdf, routes_gdf, merged_ramp_nodes_gdf, route_lines_gdf, ramp_lines_gdf, trimmed_intersections_gdf)

    # Visualize the evacuation routes, centroids, and filtered ramps
    logger.info("Visualizing the evacuation routes, centroids, and filtered ramps")
    visualize(centroids_gdf, routes_gdf, route_lines_gdf, ramp_lines_gdf, merged_ramp_nodes_gdf, trimmed_intersections_gdf, config['output_image_path'])
    logger.info(f"Visualization saved to {config['output_image_path']}")

    # Save the graph to a PNG file
    logger.info("Drawing Graph")
    draw_graph(G, config['graph_image_path'])
    logger.info(f"Graph saved to {config['graph_image_path']}")

def short_processing():
    ''' Short processing function to load pre-processed data from JSON files '''

    # Load centroids JSON
    logger.info("Loading centroids JSON")
    centroids_gdf = gpd.read_file(config['centroids_path'])

    # Load evacuation routes GeoJSON
    logger.info("Loading evacuation routes GeoJSON")
    routes_gdf = pgj.load_routes(config['routes_geojson_path'])

    # Load pre-processed GeoDataFrames from JSON files
    logger.info("Loading pre-processed GeoDataFrames from JSON files")
    merged_ramp_nodes_gdf = gpd.read_file(config['merged_ramp_nodes_path'])
    trimmed_intersections_gdf = gpd.read_file(config['trimmed_intersections_path'])

    # Load the minimum distances CSV file
    min_distances_df = pd.read_csv(config['min_distances_path'])

    # Parse the geometries from WKT format
    min_distances_df['route_lines'] = min_distances_df['route_lines'].apply(wkt.loads)
    min_distances_df['ramp_lines'] = min_distances_df['ramp_lines'].apply(wkt.loads)

    # Create GeoDataFrames for the lines
    route_lines_gdf = gpd.GeoDataFrame(min_distances_df, geometry='route_lines', crs="EPSG:3857")
    ramp_lines_gdf = gpd.GeoDataFrame(min_distances_df, geometry='ramp_lines', crs="EPSG:3857")

    # Create a graph from the GeoDataFrames
    logger.info("Creating a graph from the GeoDataFrames")
    G = create_graph(centroids_gdf, routes_gdf, merged_ramp_nodes_gdf, route_lines_gdf, ramp_lines_gdf, trimmed_intersections_gdf)

    # Visualize the evacuation routes, centroids, and filtered ramps
    logger.info("Visualizing the evacuation routes, centroids, and filtered ramps")
    visualize(centroids_gdf, routes_gdf, route_lines_gdf, ramp_lines_gdf, merged_ramp_nodes_gdf, trimmed_intersections_gdf, config['output_image_path'])
    logger.info(f"Visualization saved to {config['output_image_path']}")

    # Save the graph to a PNG file
    logger.info("Drawing Graph")
    draw_graph(G, config['graph_image_path'])
    logger.info(f"Graph saved to {config['graph_image_path']}")

def main():
    # Choose whether to run long processing or short processing
    use_long_processing = False

    if use_long_processing:
        long_processing()
    else:
        short_processing()

    # # Calculate minimum distances from centroids to evacuation routes
    # logger.info("Calculating minimum distances from centroids to evacuation routes")
    # route_min_distances, route_lines = pgdf.calculate_min_distances_to_routes(centroids_gdf, routes_gdf)

    # # Calculate minimum distances from centroids to ramps
    # logger.info("Calculating minimum distances from centroids to ramps")
    # ramp_min_distances, ramp_lines = pgdf.calculate_min_distances_to_ramps(centroids_gdf, merged_ramp_nodes_gdf, num_paths=1)

    # # Save the results to a CSV file
    # min_distances_df = pd.DataFrame(route_min_distances + ramp_min_distances)
    # min_distances_df.to_csv(config['output_csv_path'], index=False)
    # logger.info(f"Minimum distances saved to {config['output_csv_path']}")

    # # Create GeoDataFrames for the lines
    # route_lines_gdf = gpd.GeoDataFrame(geometry=route_lines, crs="EPSG:3857")
    # ramp_lines_gdf = gpd.GeoDataFrame(geometry=ramp_lines, crs="EPSG:3857")

    # # Create nodes for intersections and route points
    # logger.info("Creating nodes along routes")
    # route_nodes_gdf = pgdf.create_nodes_along_routes(routes_gdf, interval=500)

    # # Create a graph from the GeoDataFrames
    # logger.info("Creating a graph from the GeoDataFrames")
    # G = create_graph(centroids_gdf, routes_gdf, merged_ramp_nodes_gdf, route_lines_gdf, ramp_lines_gdf, trimmed_intersections_gdf, route_nodes_gdf)

    # # Save the graph to a PNG file
    # logger.info("Drawing Graph")
    # draw_graph(G, config['graph_image_path'])
    # logger.info(f"Graph saved to {config['graph_image_path']}.png")

    # # Visualize the evacuation routes, centroids, and filtered ramps
    # logger.info("Visualizing the evacuation routes, centroids, and filtered ramps")
    # visualize(centroids_gdf, routes_gdf, route_lines_gdf, ramp_lines_gdf, merged_ramp_nodes_gdf, trimmed_intersections_gdf, config['output_image_path'])
    # logger.info(f"Visualization saved to {config['output_image_path']}")

    # Perform maximum flow analysis (example)
    # source = 'some_source_node'
    # target = 'some_target_node'
    # flow_value, flow_dict = nx.maximum_flow(G, source, target)
    # logger.info(f"Maximum flow from {source} to {target}: {flow_value}")

if __name__ == "__main__":
    main()