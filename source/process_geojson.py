import geopandas as gpd
import json
from shapely.geometry import Point

from loggingFormatterEST import setup_logging

# Configure logging
logger = setup_logging('process_geojson.log', 'process_geojson')

# Loads a GeoJSON file into a GeoDataFrame.
def load_geojson(file_path):  
    return gpd.read_file(file_path)

# Loads ZIP code GeoJSON and standardizes ZIP code column
def load_zip_geojson(file_path, zip_column="Zip_Code"):
    gdf = load_geojson(file_path)
    gdf['ZipCode'] = gdf[zip_column].astype(str)
    
    # Remove the original Zip_Code column to avoid duplication
    gdf = gdf.drop(columns=[zip_column])
    
    # Check if the necessary columns are present
    required_columns = {'ZipCode', 'geometry'}
    if not required_columns.issubset(gdf.columns):
        raise ValueError(f"Missing required columns in zip_codes_gdf: {required_columns - set(gdf.columns)}")
    
    gdf['type'] = 'zip_code'
    return gdf

def load_routes(geojson_path):
    routes_gdf = gpd.read_file(geojson_path)
    routes_gdf = routes_gdf.to_crs(epsg=3857)  # Ensure the CRS matches the centroids
    return routes_gdf

def load_ramps(geojson_path):
    ramps_gdf = gpd.read_file(geojson_path)
    ramps_gdf = ramps_gdf.to_crs(epsg=3857)  # Ensure the CRS matches the centroids
    return ramps_gdf

def load_intersections(intersections_geojson_path):
    intersections_gdf = gpd.read_file(intersections_geojson_path)
    intersections_gdf = intersections_gdf.to_crs(epsg=3857)  # Ensure the CRS matches the centroids
    return intersections_gdf

def create_zip_code_centroids(geojson_path, output_json_path):
    # Load the GeoJSON file
    zip_polygons = gpd.read_file(geojson_path)
    
    # Reproject to a projected CRS for accurate centroid calculations
    zip_polygons = zip_polygons.to_crs(epsg=3857)  # Web Mercator projection
    zip_polygons['centroid'] = zip_polygons.geometry.centroid

    # Assuming the correct column name is 'Zip_Code'
    zip_centroids = {
        row['Zip_Code']: {'centroid_x': row.centroid.x, 'centroid_y': row.centroid.y}
        for _, row in zip_polygons.iterrows()
    }

    # Save the dictionary to a JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(zip_centroids, json_file, indent=4)

    logger.info(f"ZIP code centroids saved to {output_json_path}")

