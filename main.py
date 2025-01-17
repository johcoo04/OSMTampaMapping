import logging
from process_graph import create_and_simplify_graph
from process_json import load_population_data
from process_geojson import load_zip_geojson, load_routes_geojson, merge_geojson_with_population
from visualize import plot_population_and_routes

# Configure logging
logging.basicConfig(level=logging.INFO)

# File paths
zip_geojson_path = "data/Zip_Codes_Tampa.geojson"
routes_geojson_path = "data/Evacuation_Routes_Tampa.geojson"
population_json_path = "data/Tampa-zip-codes-data.json"
output_geojson_path = "data/Combined_Zip_Population_Routes.geojson"

def main():
    try:
        # Step 1: Load population data
        population_df = load_population_data(population_json_path)
        logging.info("Population data loaded successfully.")
        
        # Step 2: Load ZIP code and route GeoJSON files
        zip_gdf = load_zip_geojson(zip_geojson_path)
        routes_gdf = load_routes_geojson(routes_geojson_path)
        logging.info("GeoJSON files loaded successfully.")
        
        # Step 3: Merge population data with ZIP codes
        combined_gdf = merge_geojson_with_population(zip_gdf, population_df)
        
        # Step 4: Save combined GeoJSON
        combined_gdf.to_file(output_geojson_path, driver="GeoJSON")
        logging.info(f"Combined GeoJSON saved to {output_geojson_path}")
        
        # Step 5: Simplify evacuation routes graph
        simplified_graph = create_and_simplify_graph(routes_geojson_path)
        logging.info("Evacuation routes graph simplified successfully.")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()