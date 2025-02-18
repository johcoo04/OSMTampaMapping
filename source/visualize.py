import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx

def visualize(centroids_gdf, routes_gdf, route_lines_gdf, ramp_lines_gdf, merged_ramp_nodes_gdf, trimmed_intersections_gdf, output_image_path):

    # Plot the GeoDataFrames
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
    trimmed_intersections_gdf.plot(ax=ax, color='magenta', markersize=5, label='Filtered Intersections')
    
    # Add ZIP code labels
    for centroid in centroids_gdf.itertuples():
        ax.annotate(f"{centroid.ZipCode}", xy=(centroid.geometry.x, centroid.geometry.y), xytext=(3, 3), textcoords="offset points", fontsize=8, color='black')
    
    plt.legend()
    plt.title("Evacuation Routes, Centroids, Shortest Path to Ramps and Intersections")
    plt.savefig(output_image_path)
    plt.show()

def draw_graph(G, output_path):
    # Extract positions from node attributes
    pos = nx.get_node_attributes(G, 'pos')

    # Separate nodes by type
    origin_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'origin']
    intersection_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'intersection']
    destination_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'destination']

    plt.figure(figsize=(10, 10))

    # Draw nodes with different colors based on type
    nx.draw_networkx_nodes(G, pos, nodelist=origin_nodes, node_size=50, node_color='blue', alpha=0.7, label='Centroids')
    nx.draw_networkx_nodes(G, pos, nodelist=intersection_nodes, node_size=50, node_color='red', alpha=0.7, label='Intersections')
    nx.draw_networkx_nodes(G, pos, nodelist=destination_nodes, node_size=50, node_color='green', alpha=0.7, label='Ramps')

    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_path}")
    plt.close()