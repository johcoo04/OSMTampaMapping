import geopandas as gpd
import matplotlib.pyplot as plt

def visualize(centroids_gdf, routes_gdf, route_lines_gdf, ramp_lines_gdf, merged_ramp_nodes_gdf, merged_intersection_nodes_gdf, output_image_path):

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
    route_lines_gdf.plot(ax=ax, color='green', linewidth=1, linestyle='--', label='Shortest Paths to Routes')
    ramp_lines_gdf.plot(ax=ax, color='orange', linewidth=1, linestyle='--', label='Shortest Paths to Ramps')
    merged_ramp_nodes_gdf.plot(ax=ax, color='cyan', markersize=5, label='Filtered Ramps')
    merged_intersection_nodes_gdf.plot(ax=ax, color='magenta', markersize=5, label='Filtered Intersections')
    
    # Add ZIP code labels
    for centroid in centroids_gdf.itertuples():
        ax.annotate(f"{centroid.ZipCode}", xy=(centroid.geometry.x, centroid.geometry.y), xytext=(3, 3), textcoords="offset points", fontsize=8, color='black')
    
    plt.legend()
    plt.title("Evacuation Routes, Centroids, Shortest Path to Ramps and Intersections")
    plt.savefig(output_image_path)
    plt.show()