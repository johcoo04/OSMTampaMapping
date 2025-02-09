import logging
import pandas as pd
import geopandas as gpd
from process_geojson import load_centroids, load_routes, load_ramps
from data_processing import calculate_min_distances, filter_ramps_within_distance, merge_close_nodes
from visualize import visualize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test log message
logger.info("Logging setup complete.")

# File paths
zip_geojson_path = "data/Zip_Codes_Tampa.geojson"
routes_geojson_path = "data/Evacuation_Routes_Tampa.geojson"
output_json_path = "data/zip_code_centroids.json"
output_csv_path = "data/min_distances.csv"
output_image_path = "data/evacuation_routes_with_centroids.png"
ramps_geojson_path = "data/Ramps_TDA.geojson"

# Commented out since data is not changing
# Load Zip GeoJSON Data
# logger.info("Reads in GeoJSON data")
# zip_gdf = load_zip_geojson(zip_geojson_path)

# Create ZIP code centroids and save to JSON
# logger.info("Creating ZIP code centroids and saving to JSON")
# create_zip_code_centroids(zip_geojson_path, output_json_path)

# Load centroids JSON
logger.info("Loading centroids JSON")
centroids_gdf = load_centroids(output_json_path)

# Load evacuation routes GeoJSON
logger.info("Loading evacuation routes GeoJSON")
routes_gdf = load_routes(routes_geojson_path)

# Load ramps GeoJSON
logger.info("Loading ramps GeoJSON")
ramps_gdf = load_ramps(ramps_geojson_path)

# Calculate minimum distances
logger.info("Calculating minimum distances from centroids to evacuation routes")
min_distances, lines = calculate_min_distances(centroids_gdf, routes_gdf)

# Save the results to a CSV file
min_distances_df = pd.DataFrame(min_distances)
min_distances_df.to_csv(output_csv_path, index=False)
logger.info(f"Minimum distances saved to {output_csv_path}")

# Create a GeoDataFrame for the lines
lines_gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:3857")

# Filter ramps within 500 meters of evacuation routes
logger.info("Filtering ramps within 500 meters of evacuation routes")
filtered_ramps_gdf = filter_ramps_within_distance(ramps_gdf, routes_gdf, distance=500)

# Merge close nodes within 50 meters
logger.info("Merging close nodes within 50 meters")
merged_nodes_gdf = merge_close_nodes(filtered_ramps_gdf, distance=50)

# Visualize the evacuation routes, centroids, and filtered ramps
logger.info("Visualizing the evacuation routes, centroids, and filtered ramps")
visualize(centroids_gdf, routes_gdf, lines_gdf, merged_nodes_gdf, output_image_path)
logger.info(f"Visualization saved to {output_image_path}")