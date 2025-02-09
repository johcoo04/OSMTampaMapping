import logging
import pandas as pd
import geopandas as gpd
from data_processing import load_centroids, load_routes, calculate_min_distances
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
output_image_path = "data/evacuation_routes_with_centroids_and_ramps.png"
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

# Calculate minimum distances
logger.info("Calculating minimum distances from centroids to evacuation routes")
min_distances, lines = calculate_min_distances(centroids_gdf, routes_gdf)

# Save the results to a CSV file
min_distances_df = pd.DataFrame(min_distances)
min_distances_df.to_csv(output_csv_path, index=False)
logger.info(f"Minimum distances saved to {output_csv_path}")

# Create a GeoDataFrame for the lines
lines_gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:3857")

# Visualize the evacuation routes, centroids, and shortest paths with ramps
logger.info("Visualizing the evacuation routes, centroids, and shortest paths with ramps")
visualize(centroids_gdf, routes_gdf, lines_gdf, output_image_path)
logger.info(f"Visualization saved to {output_image_path}")