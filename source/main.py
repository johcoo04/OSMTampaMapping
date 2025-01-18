from loggingFormatterEST import setup_logging
from process_geojson import combine_geojson_data, save_combined_geojson_to_csv
from process_graph import combine_data_to_graph
from visualize import visualize_combined_graph
import os

# Configure logging
logger = setup_logging('main.log','main')

# File paths
zip_geojson_path = "data/Zip_Codes_Tampa.geojson"
routes_geojson_path = "data/Evacuation_Routes_Tampa.geojson"
output_csv_path = "geodataframe.csv"

# Combine ZIP code and route data into a single GeoDataFrame
logger.info("Combining ZIP code and route data into a single GeoDataFrame")
combined_gdf = combine_geojson_data(zip_geojson_path, routes_geojson_path)

# Save the combined GeoDataFrame to a CSV file
logger.info("Saving the combined GeoDataFrame to a CSV file")
save_combined_geojson_to_csv(combined_gdf, output_csv_path)

# Verify if the CSV file was created
if os.path.exists(output_csv_path):
    logger.info(f"CSV file created successfully at {output_csv_path}")
else:
    logger.error(f"Failed to create CSV file at {output_csv_path}")

# Combine ZIP nodes with routes
logger.info("Combining ZIP nodes with routes")
combined_graph = combine_data_to_graph(combined_gdf)

# Visualize the graph
logger.info("Visualizing the graph")
#visualize_combined_graph(combined_graph)