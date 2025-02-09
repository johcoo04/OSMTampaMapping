import geopandas as gpd
import json
from shapely.geometry import Point, LineString

def load_centroids(json_path):
    with open(json_path, 'r') as f:
        centroids_data = json.load(f)
    
    centroids_list = [
        {'ZipCode': zip_code, 'geometry': Point(data['centroid_x'], data['centroid_y'])}
        for zip_code, data in centroids_data.items()
    ]
    
    centroids_gdf = gpd.GeoDataFrame(centroids_list, crs="EPSG:3857")
    return centroids_gdf

def load_routes(geojson_path):
    routes_gdf = gpd.read_file(geojson_path)
    routes_gdf = routes_gdf.to_crs(epsg=3857)  # Ensure the CRS matches the centroids
    return routes_gdf

def calculate_min_distances(centroids_gdf, routes_gdf):
    min_distances = []
    lines = []
    
    for centroid in centroids_gdf.itertuples():
        distances = routes_gdf.distance(centroid.geometry)
        min_distance = distances.min()
        nearest_route = routes_gdf.iloc[distances.idxmin()]
        nearest_point = nearest_route.geometry.interpolate(nearest_route.geometry.project(centroid.geometry))
        min_distances.append({'ZipCode': centroid.ZipCode, 'min_distance': min_distance, 'nearest_point': nearest_point})
        lines.append(LineString([centroid.geometry, nearest_point]))
    
    return min_distances, lines