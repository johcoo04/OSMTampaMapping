import geopandas as gpd
import matplotlib.pyplot as plt
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

def visualize(zip_geojson_path, routes_geojson_path, centroids_json_path, output_image_path):
    # Load data
    centroids_gdf = load_centroids(centroids_json_path)
    routes_gdf = load_routes(routes_geojson_path)
    
    # Calculate minimum distances and lines
    min_distances, lines = calculate_min_distances(centroids_gdf, routes_gdf)
    
    # Create a GeoDataFrame for the lines
    lines_gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:3857")
    
    # Plot the data
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Define colors for each road type
    road_colors = {
        'C': 'blue',
        'H': 'red',
        'M': 'green',
        'P': 'purple'
    }
    
    # Plot each road type with a different color
    for road_type, color in road_colors.items():
        routes_gdf[routes_gdf['TYPE'] == road_type].plot(ax=ax, color=color, linewidth=1, label=f'Road Type {road_type}')
    
    centroids_gdf.plot(ax=ax, color='black', markersize=5, label='Centroids')
    lines_gdf.plot(ax=ax, color='orange', linewidth=1, linestyle='--', label='Shortest Paths')
    
    # Add ZIP code labels
    for centroid in centroids_gdf.itertuples():
        ax.annotate(f"{centroid.ZipCode}", xy=(centroid.geometry.x, centroid.geometry.y), xytext=(3, 3), textcoords="offset points", fontsize=8, color='black')
    
    plt.legend()
    plt.title("Evacuation Routes, Centroids, and Shortest Paths")
    plt.savefig(output_image_path)
    plt.show()

# Example usage
if __name__ == "__main__":
    zip_geojson_path = "data/Zip_Codes_Tampa.geojson"
    routes_geojson_path = "data/Evacuation_Routes_Tampa.geojson"
    centroids_json_path = "data/zip_code_centroids.json"
    output_image_path = "data/evacuation_routes_with_centroids.png"
    
    visualize(zip_geojson_path, routes_geojson_path, centroids_json_path, output_image_path)
