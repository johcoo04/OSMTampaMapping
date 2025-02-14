import logging
import pandas as pd
import geopandas as gpd
import numpy as np
from process_json import load_centroids, load_population_data
import process_geojson as pgj
import process_geoDataFrames as pgdf
from visualize import visualize
from create_graph import create_graph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File paths
zip_geojson_path = "data/Zip_Codes_Tampa.geojson"
routes_geojson_path = "data/Evacuation_Routes_Tampa.geojson"
output_json_path = "data/zip_code_centroids.json"
output_csv_path = "data/min_distances.csv"
output_image_path = "data/evacuation_routes_with_centroids.png"
ramps_geojson_path = "data/Ramps_TDA.geojson"
intersections_geojson_path = "data/Intersection.geojson"

def main():

    ''' Commented out since data is not changing often'''
    # Load Zip GeoJSON Data
    # logger.info("Reads in GeoJSON data")
    # zip_gdf = pgj.load_zip_geojson(zip_geojson_path)

    # Create ZIP code centroids and save to JSON
    # logger.info("Creating ZIP code centroids and saving to JSON")
    # pgj.create_zip_code_centroids(zip_geojson_path, output_json_path)

    # Load centroids JSON
    logger.info("Loading centroids JSON")
    centroids_gdf = load_centroids(output_json_path)

    # Load evacuation routes GeoJSON
    logger.info("Loading evacuation routes GeoJSON")
    routes_gdf = pgj.load_routes(routes_geojson_path)

    # Load ramps GeoJSON
    logger.info("Loading ramps GeoJSON")
    ramps_gdf = pgj.load_ramps(ramps_geojson_path)

    # Filter ramps near highways
    logger.info("Filtering ramps within 500 meters of the highways")
    filtered_ramps_gdf = pgdf.filter_ramps_near_highways(ramps_gdf, routes_gdf, distance=500)

    # Merge close nodes within 50 meters
    logger.info("Merging close nodes within 500 meters")
    merged_ramp_nodes_gdf = pgdf.merge_close_nodes(filtered_ramps_gdf, distance=500)

    # Load intersections GeoJSON
    logger.info("Loading intersections GeoJSON")
    intersections_gdf = pgj.load_intersections(intersections_geojson_path)

    # Filter intersections near evacuation routes
    logger.info("Filtering intersections within 50 meters of evacuation routes")
    filtered_intersections_gdf = pgdf.filter_intersections_near_route(intersections_gdf, routes_gdf, distance=50)

    # Merge close intersections within 500 meters
    logger.info("Merging close intersections within 500 meters")
    merged_intersection_nodes_gdf = pgdf.merge_close_nodes(filtered_intersections_gdf, distance=500)

    # Calculate minimum distances from centroids to evacuation routes
    logger.info("Calculating minimum distances from centroids to evacuation routes")
    route_min_distances, route_lines = pgdf.calculate_min_distances_to_routes(centroids_gdf, routes_gdf)

    # Calculate minimum distances from centroids to ramps
    logger.info("Calculating minimum distances from centroids to ramps")
    ramp_min_distances, ramp_lines = pgdf.calculate_min_distances_to_ramps(centroids_gdf, merged_ramp_nodes_gdf, num_paths=1)

    # Save the results to a CSV file
    min_distances_df = pd.DataFrame(route_min_distances + ramp_min_distances)
    min_distances_df.to_csv(output_csv_path, index=False)
    logger.info(f"Minimum distances saved to {output_csv_path}")

    # Create GeoDataFrames for the lines
    route_lines_gdf = gpd.GeoDataFrame(geometry=route_lines, crs="EPSG:3857")
    ramp_lines_gdf = gpd.GeoDataFrame(geometry=ramp_lines, crs="EPSG:3857")

    # Create nodes for intersections and route points
    # intersection_nodes_gdf = pgdf.create_nodes_along_routes(routes_gdf, interval=500)
    #route_nodes_gdf = pgdf.create_nodes_along_routes(routes_gdf, interval=500)

    # Visualize the evacuation routes, centroids, and filtered ramps
    logger.info("Visualizing the evacuation routes, centroids, and filtered ramps")
    # visualize(centroids_gdf, routes_gdf, route_lines_gdf, ramp_lines_gdf, merged_ramp_nodes_gdf, merged_intersection_nodes_gdf, route_nodes_gdf, output_image_path)
    visualize(centroids_gdf, routes_gdf, route_lines_gdf, ramp_lines_gdf, merged_ramp_nodes_gdf, merged_intersection_nodes_gdf, output_image_path)
    logger.info(f"Visualization saved to {output_image_path}")

    # # Create a graph from the GeoDataFrames
    # logger.info("Creating a graph from the GeoDataFrames")
    # G = create_graph(centroids_gdf, routes_gdf, merged_ramp_nodes_gdf, route_lines_gdf, ramp_lines_gdf, merged_intersection_nodes_gdf, route_nodes_gdf)

    # # Perform maximum flow analysis (example)
    # source = 'some_source_node'
    # target = 'some_target_node'
    # flow_value, flow_dict = nx.maximum_flow(G, source, target)
    # logger.info(f"Maximum flow from {source} to {target}: {flow_value}")

if __name__ == "__main__":
    main()