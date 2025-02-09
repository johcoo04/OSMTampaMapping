import geopandas as gpd
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_zip_code_centroids(geojson_path, output_json_path):
    # Load the GeoJSON file
    zip_polygons = gpd.read_file(geojson_path)
    
    # Reproject to a projected CRS for accurate centroid calculations
    zip_polygons = zip_polygons.to_crs(epsg=3857)  # Web Mercator projection
    zip_polygons['centroid'] = zip_polygons.geometry.centroid

    # Create a dictionary to store ZIP code centroids
    zip_centroids = {
        row['Zip_Code']: {'centroid_x': row.centroid.x, 'centroid_y': row.centroid.y}
        for _, row in zip_polygons.iterrows()
    }

    # Save the dictionary to a JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(zip_centroids, json_file, indent=4)

    logger.info(f"ZIP code centroids saved to {output_json_path}")

# Example usage
if __name__ == "__main__":
    geojson_path = "data/Zip_Codes_Tampa.geojson"
    output_json_path = "data/zip_code_centroids.json"
    create_zip_code_centroids(geojson_path, output_json_path)