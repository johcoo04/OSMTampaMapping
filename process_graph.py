import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString

# Simplify a graph by merging intermediate nodes that form straight lines.
def simplify_graph_topology(gdf):
    simplified_graph = nx.Graph()

    for _, row in gdf.iterrows():
        # Convert the road geometry to a LineString
        line = row.geometry

        # Add all points from the LineString
        coords = list(line.coords)

        # Add nodes and edges, but merge intermediate nodes
        for i in range(len(coords) - 1):
            start = coords[i]
            end = coords[i + 1]

            # Add the start and end points as nodes
            if not simplified_graph.has_node(start):
                simplified_graph.add_node(start, pos=start)
            if not simplified_graph.has_node(end):
                simplified_graph.add_node(end, pos=end)

            # Add an edge between the start and end points
            simplified_graph.add_edge(start, end, weight=LineString([start, end]).length)

    # Merge intermediate nodes
    nodes_to_remove = []
    for node in list(simplified_graph.nodes):
        neighbors = list(simplified_graph.neighbors(node))

        # If a node has exactly 2 neighbors (not an endpoint or intersection)
        if len(neighbors) == 2:
            # Check if the neighbors form a straight line
            n1, n2 = neighbors
            v1 = (node[0] - n1[0], node[1] - n1[1])  # Vector from n1 to node
            v2 = (n2[0] - node[0], n2[1] - node[1])  # Vector from node to n2

            # Compute the cross product to check for collinearity
            cross_product = abs(v1[0] * v2[1] - v1[1] * v2[0])

            if cross_product < 1e-6:  # Collinear
                # Merge the two edges into one and remove the intermediate node
                new_edge_length = simplified_graph[n1][node]['weight'] + simplified_graph[node][n2]['weight']
                simplified_graph.add_edge(n1, n2, weight=new_edge_length)
                nodes_to_remove.append(node)

    # Remove unnecessary intermediate nodes
    simplified_graph.remove_nodes_from(nodes_to_remove)

    return simplified_graph

# Create and simplify a graph from a GeoJSON file.
def create_and_simplify_graph(file_path):
    gdf = gpd.read_file(file_path)
    return simplify_graph_topology(gdf)
