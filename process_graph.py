import networkx as nx
from shapely.geometry import LineString

def create_zip_nodes(zip_codes_gdf):
    # Convert ZIP codes to dictionary of nodes (centroids)
    zip_nodes = {
        row['ZipCode']: (row.geometry.centroid.x, row.geometry.centroid.y)
        for _, row in zip_codes_gdf.iterrows()
    }
    return zip_nodes

def add_zip_nodes(graph, zip_nodes):
    for zip_code, coords in zip_nodes.items():
        graph.add_node(zip_code, pos=coords, type='zip_code')


def add_route_nodes_and_edges(graph, routes_gdf):

    for _, row in routes_gdf.iterrows():
        line = row.geometry
        coords = list(line.coords)

        for i in range(len(coords) - 1):
            start = coords[i]
            end = coords[i + 1]

            if not graph.has_node(start):
                graph.add_node(start, pos=start, type='route')
            if not graph.has_node(end):
                graph.add_node(end, pos=end, type='route')

            graph.add_edge(start, end, weight=LineString([start, end]).length)


def simplify_graph(graph):
    nodes_to_remove = []
    for node in list(graph.nodes):
        if graph.nodes[node]['type'] == 'route':
            neighbors = list(graph.neighbors(node))
            if len(neighbors) == 2:
                n1, n2 = neighbors
                v1 = (node[0] - n1[0], node[1] - n1[1])
                v2 = (n2[0] - node[0], n2[1] - node[1])

                cross_product = abs(v1[0] * v2[1] - v1[1] * v2[0])
                if cross_product < 1e-6:
                    new_edge_length = (
                        graph[n1][node]['weight'] + graph[node][n2]['weight']
                    )
                    graph.add_edge(n1, n2, weight=new_edge_length)
                    nodes_to_remove.append(node)

    graph.remove_nodes_from(nodes_to_remove)


def combine_data_to_graph(zip_codes_gdf, routes_gdf):
    graph = nx.Graph()
    zip_nodes = create_zip_nodes(zip_codes_gdf)
    add_zip_nodes(graph, zip_nodes)
    add_route_nodes_and_edges(graph, routes_gdf)
    simplify_graph(graph)
    return graph

