import geopandas as gpd
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

def create_zip_code_centroids(zip_geojson_path, output_path):
    ''' Create ZIP code centroids and save to JSON '''
    zip_gdf = gpd.read_file(zip_geojson_path)
    
    # Ensure the CRS is projected for accurate centroid calculations
    zip_gdf = zip_gdf.to_crs(epsg=3857)
    
    # Calculate centroids
    zip_gdf['centroid'] = zip_gdf.geometry.centroid
    
    # Ensure the ZipCode column exists
    if 'ZipCode' not in zip_gdf.columns:
        zip_gdf['ZipCode'] = zip_gdf['Zip_Code'].astype(str)
    
    # List of ZIP codes to remove
    zip_codes_to_remove = [
        "33715", "33711", "34221", "34219", "33834", "33860", "33811", "33815", "33810", "33849",
        "33540", "33541", "33543", "33544", "34638", "34655", "34688", "34685", "34677", "33759"
    ]
    
    # Filter out the specified ZIP codes
    zip_gdf = zip_gdf[~zip_gdf['ZipCode'].isin(zip_codes_to_remove)]
    
    centroids_gdf = zip_gdf[['ZipCode', 'centroid']].copy()
    centroids_gdf = centroids_gdf.rename(columns={'centroid': 'geometry'})
    centroids_gdf = gpd.GeoDataFrame(centroids_gdf, geometry='geometry')
    centroids_gdf.to_file(output_path, driver="GeoJSON")
    return centroids_gdf

