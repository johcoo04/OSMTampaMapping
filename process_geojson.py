import geopandas as gpd

def load_zip_geojson(geojson_path):
    """
    Load ZIP code GeoJSON data.

    Args:
        geojson_path (str): Path to the ZIP code GeoJSON file.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame of ZIP code data.
    """
    # Load GeoJSON data into a GeoDataFrame
    gdf = gpd.read_file(geojson_path)
    
    # Convert 'Zip_Code' column to string and rename it to 'ZipCode'
    gdf['ZipCode'] = gdf['Zip_Code'].astype(str)
    
    return gdf

def load_routes_geojson(geojson_path):
    """
    Load evacuation routes GeoJSON data.

    Args:
        geojson_path (str): Path to the evacuation routes GeoJSON file.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame of evacuation routes data.
    """
    # Load evacuation routes GeoJSON
    return gpd.read_file(geojson_path)

def merge_geojson_with_population(zip_gdf, population_df):
    """
    Merge ZIP code GeoJSON data with population data.

    Args:
        zip_gdf (gpd.GeoDataFrame): GeoDataFrame of ZIP code data.
        population_df (pd.DataFrame): DataFrame of population data.

    Returns:
        gpd.GeoDataFrame: Merged GeoDataFrame.
    """
    # Merge population data into the ZIP GeoDataFrame on 'ZipCode'
    merged_gdf = zip_gdf.merge(population_df, on='ZipCode', how='left')
    return merged_gdf
