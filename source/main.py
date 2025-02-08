import logging
import os
from process_geojson import load_zip_geojson, load_centroids, load_routes, calculate_min_distances
from create_centroids import create_zip_code_centroids
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

# Load Zip GeoJSON Data
logger.info("Reads in GeoJSON data")
zip_gdf = load_zip_geojson(zip_geojson_path)

# Create ZIP code centroids and save to JSON
logger.info("Creating ZIP code centroids and saving to JSON")
create_zip_code_centroids(zip_geojson_path, output_json_path)

# Load centroids JSON
logger.info("Loading centroids JSON")
centroids_gdf = load_centroids(output_json_path)

# Load evacuation routes GeoJSON
logger.info("Loading evacuation routes GeoJSON")
routes_gdf = load_routes(routes_geojson_path)

# Calculate minimum distances
logger.info("Calculating minimum distances from centroids to evacuation routes")
min_distances_df = calculate_min_distances(centroids_gdf, routes_gdf)

# Save the results to a CSV file
min_distances_df.to_csv(output_csv_path, index=False)
logger.info(f"Minimum distances saved to {output_csv_path}")

# Visualize the evacuation routes, centroids, and shortest paths
logger.info("Visualizing the evacuation routes, centroids, and shortest paths")
visualize(zip_geojson_path, routes_geojson_path, output_json_path, output_image_path)
logger.info(f"Visualization saved to {output_image_path}")