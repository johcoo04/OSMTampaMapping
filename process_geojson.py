import geopandas as gpd

#Load ZIP code GeoJSON data.
def load_zip_geojson(geojson_path):
    gdf = gpd.read_file(geojson_path)
    
    # Convert 'Zip_Code' column to string and rename it to 'ZipCode'
    gdf['ZipCode'] = gdf['Zip_Code'].astype(str)
    
    return gdf

#Load evacuation routes GeoJSON data.
def load_routes_geojson(geojson_path):
    return gpd.read_file(geojson_path)

# Merge ZIP code GeoJSON data with population data.
def merge_geojson_with_population(zip_gdf, population_df):
    merged_gdf = zip_gdf.merge(population_df, on='ZipCode', how='left')
    return merged_gdf
