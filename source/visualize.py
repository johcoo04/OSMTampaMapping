import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx

def visualize_zip_codes_with_graph(geojson_path, json_data_path):
    # Load the GeoJSON file
    zip_polygons = gpd.read_file(geojson_path)
    print(f"GeoJSON Columns: {zip_polygons.columns}")

    # Reproject to a projected CRS for accurate centroid calculations
    zip_polygons = zip_polygons.to_crs(epsg=3857)  # Web Mercator projection
    zip_polygons['centroid'] = zip_polygons.geometry.centroid

    # Add centroids to the GeoDataFrame
    zip_polygons['centroid_x'] = zip_polygons.centroid.x
    zip_polygons['centroid_y'] = zip_polygons.centroid.y

    # Initialize a graph
    graph = nx.Graph()

    # Add ZIP code nodes with centroid positions
    for _, row in zip_polygons.iterrows():
        graph.add_node(row['Zip_Code'], pos=(row['centroid_x'], row['centroid_y']))

    # Add edges for adjacent ZIP codes
    for idx1, row1 in zip_polygons.iterrows():
        for idx2, row2 in zip_polygons.iterrows():
            if idx1 != idx2 and row1.geometry.touches(row2.geometry):
                graph.add_edge(row1['Zip_Code'], row2['Zip_Code'])

    # Plot the ZIP code polygons
    fig, ax = plt.subplots(figsize=(12, 12))
    zip_polygons.plot(ax=ax, color='lightblue', edgecolor='black', alpha=0.5)

    # Plot the graph on top of the polygons
    pos = nx.get_node_attributes(graph, 'pos')
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=50,
        node_color='red',
        font_size=8,
        edge_color='gray',
        ax=ax,
    )

    # Add a title and save the visualization
    plt.title("ZIP Code Polygons and Adjacency Graph")
    plt.savefig("zip_code_graph_visualization.png")
    plt.show()

# Example usage
if __name__ == "__main__":
    visualize_zip_codes_with_graph(
        geojson_path="data/Zip_Codes_Tampa.geojson",
        json_data_path="data/Tampa-zip-codes-data.json"
    )
