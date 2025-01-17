import geopandas as gpd

# Loads a GeoJSON file into a GeoDataFrame.
def load_geojson(file_path):  
    return gpd.read_file(file_path)

# Loads ZIP code GeoJSON and Standardizes ZIP code column
def load_zip_geojson(file_path, zip_column="Zip_Code"):
    gdf = load_geojson(file_path)
    gdf['ZipCode'] = gdf[zip_column].astype(str)  
    return gdf

# Loads routes GeoJSON.
def load_routes_geojson(file_path):
    return load_geojson(file_path)

# May not be need but loads Shelter locations and creates standard
# shelter ID column
def load_shelters_geojson(file_path, shelter_id_column="ID"):
    gdf = load_geojson(file_path)
    gdf['ShelterID'] = gdf[shelter_id_column] 
    return gdf
