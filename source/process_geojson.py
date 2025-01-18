import logging
import geopandas as gpd
import pandas as pd

from inspect_csv import inspect_csv
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

# Loads routes GeoJSON.
def load_routes_geojson(file_path):
    gdf = load_geojson(file_path)
    
    # Check if the necessary columns are present
    required_columns = {'geometry'}
    if not required_columns.issubset(gdf.columns):
        raise ValueError(f"Missing required columns in routes_gdf: {required_columns - set(gdf.columns)}")
    
    gdf['type'] = 'route'
    return gdf

# Combine ZIP code and route data into a single GeoDataFrame
def combine_geojson_data(zip_geojson_path, routes_geojson_path):
    zip_codes_gdf = load_zip_geojson(zip_geojson_path)
    routes_gdf = load_routes_geojson(routes_geojson_path)
    combined_gdf = gpd.GeoDataFrame(pd.concat([zip_codes_gdf, routes_gdf], ignore_index=True))
    return combined_gdf

# Save the combined GeoDataFrame to a CSV file
def save_combined_geojson_to_csv(combined_gdf, output_csv_path):
    combined_gdf.to_csv(output_csv_path, index=False)
    inspect_csv(output_csv_path)

