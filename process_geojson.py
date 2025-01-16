import geopandas as gpd

def load_geojson(file_path):
    return gpd.read_file(file_path)

def load_zip_geojson(file_path, zip_column="Zip_Code"):
    gdf = load_geojson(file_path)
    gdf['ZipCode'] = gdf[zip_column].astype(str)  # Standardize ZIP code column
    return gdf

def load_routes_geojson(file_path):
    return load_geojson(file_path)

def load_shelters_geojson(file_path, shelter_id_column="ID"):
    gdf = load_geojson(file_path)
    gdf['ShelterID'] = gdf[shelter_id_column]  # Rename and standardize shelter ID column
    return gdf
