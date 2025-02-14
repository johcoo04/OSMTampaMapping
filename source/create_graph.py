import networkx as nx
import geopandas as gpd

def create_graph(centroids_gdf, routes_gdf, merged_nodes_gdf, route_lines_gdf, ramp_lines_gdf, intersection_nodes_gdf, route_nodes_gdf):
    G = nx.Graph()

    # Add nodes for centroids
    for idx, row in centroids_gdf.iterrows():
        G.add_node(row['ZipCode'], pos=(row.geometry.x, row.geometry.y), type='centroid')

    # Add nodes for intersections
    for idx, row in intersection_nodes_gdf.iterrows():
        G.add_node(f"intersection_{idx}", pos=(row.geometry.x, row.geometry.y), type='intersection')

    # Add nodes for route nodes
    for idx, row in route_nodes_gdf.iterrows():
        G.add_node(f"route_{idx}", pos=(row.geometry.x, row.geometry.y), type='route')

    # Add nodes for merged ramps
    for idx, row in merged_nodes_gdf.iterrows():
        G.add_node(f"ramp_{idx}", pos=(row.geometry.x, row.geometry.y), type='ramp')

    # Add edges for the shortest paths to routes
    for line in route_lines_gdf.geometry:
        start = (line.coords[0][0], line.coords[0][1])
        end = (line.coords[1][0], line.coords[1][1])
        G.add_edge(start, end, weight=line.length)

    # Add edges for the shortest paths to ramps
    for line in ramp_lines_gdf.geometry:
        start = (line.coords[0][0], line.coords[0][1])
        end = (line.coords[1][0], line.coords[1][1])
        G.add_edge(start, end, weight=line.length)

    return G