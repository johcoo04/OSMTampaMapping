import networkx as nx
from shapely.geometry import LineString
import geopandas as gpd
from loggingFormatterEST import setup_logging

# Configure logging
logger = setup_logging('process_graph.log', 'process_graph')

# Converts ZIP codes to a dictionary of nodes (centroids).
def create_zip_nodes(zip_codes_gdf):
    zip_nodes = {
        row['ZipCode']: (row.geometry.centroid.x, row.geometry.centroid.y)
        for _, row in zip_codes_gdf.iterrows()
    }
    logger.debug("Created ZIP nodes: %s", zip_nodes)
    return zip_nodes

# Adds ZIP code nodes to the graph.
def add_zip_nodes(graph, zip_nodes):
    for zip_code, coords in zip_nodes.items():
        graph.add_node(zip_code, pos=coords, type='zip_code')
    logger.debug("Added ZIP nodes to graph: %s", graph.nodes(data=True))

# Adds route nodes and edges to the graph.
def add_route_nodes_and_edges(graph, routes_gdf):
    for _, row in routes_gdf.iterrows():
        line = row.geometry
        coords = list(line.coords)
        road_type = row['TYPE']  # Get the road type

        for i in range(len(coords) - 1):
            start = f"route_{coords[i]}"
            end = f"route_{coords[i + 1]}"
            start_pos = coords[i]
            end_pos = coords[i + 1]

            if not graph.has_node(start):
                graph.add_node(start, pos=start_pos, type='route')
            if not graph.has_node(end):
                graph.add_node(end, pos=end_pos, type='route')

            graph.add_edge(start, end, weight=LineString([start_pos, end_pos]).length, road_type=road_type)
    logger.debug("Added route nodes and edges to graph: %s", graph.edges(data=True))

# Simplifies the graph by removing unnecessary nodes.
def simplify_graph(graph):
    nodes_to_remove = []
    for node in list(graph.nodes):
        if graph.nodes[node]['type'] == 'route':
            neighbors = list(graph.neighbors(node))
            if len(neighbors) == 2:
                n1, n2 = neighbors
                v1 = (
                    graph.nodes[node]['pos'][0] - graph.nodes[n1]['pos'][0],
                    graph.nodes[node]['pos'][1] - graph.nodes[n1]['pos'][1]
                )
                v2 = (
                    graph.nodes[n2]['pos'][0] - graph.nodes[node]['pos'][0],
                    graph.nodes[n2]['pos'][1] - graph.nodes[node]['pos'][1]
                )

                cross_product = abs(v1[0] * v2[1] - v1[1] * v2[0])
                if cross_product < 1e-6:
                    new_edge_length = (
                        graph[n1][node]['weight'] + graph[node][n2]['weight']
                    )
                    graph.add_edge(n1, n2, weight=new_edge_length)
                    nodes_to_remove.append(node)
    graph.remove_nodes_from(nodes_to_remove)
    logger.debug("Simplified graph: %s", graph.nodes(data=True))

# Combines ZIP code and route data into a graph.
def combine_data_to_graph(combined_gdf):
    zip_codes_gdf = combined_gdf[combined_gdf['type'] == 'zip_code']
    routes_gdf = combined_gdf[combined_gdf['type'] == 'route']
    
    graph = nx.Graph()
    zip_nodes = create_zip_nodes(zip_codes_gdf)
    add_zip_nodes(graph, zip_nodes)
    add_route_nodes_and_edges(graph, routes_gdf)
    simplify_graph(graph)
    logger.debug("Combined graph: %s", graph.nodes(data=True))
    return graph
