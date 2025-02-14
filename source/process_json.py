import pandas as pd
import json as json
import geopandas as gpd
from shapely.geometry import Point

# Load and prepare population data from a JSON file.
def load_population_data(json_path):
    population_df = pd.read_json(json_path)
    population_df.rename(columns={'zip': 'ZipCode', 'population': 'Population'}, inplace=True)
    population_df['ZipCode'] = population_df['ZipCode'].astype(str)
    return population_df

def load_centroids(json_path):
    with open(json_path, 'r') as f:
        centroids_data = json.load(f)
    
    centroids_list = [
        {'ZipCode': zip_code, 'geometry': Point(data['centroid_x'], data['centroid_y'])}
        for zip_code, data in centroids_data.items()
    ]
    
    centroids_gdf = gpd.GeoDataFrame(centroids_list, crs="EPSG:3857")
    return centroids_gdf