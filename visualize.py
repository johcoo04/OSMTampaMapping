import matplotlib.pyplot as plt
import networkx as nx

def visualize_combined_graph(graph):
    """
    Visualize the combined graph with ZIP code nodes and routes.

    Args:
        graph (nx.Graph): Combined graph of ZIP code and route nodes.
    """
    pos = nx.get_node_attributes(graph, 'pos')

    # Separate ZIP code nodes and route nodes
    zip_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'zip_code']
    route_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'route']

    # Plot ZIP code nodes
    zip_positions = {n: pos[n] for n in zip_nodes}
    nx.draw_networkx_nodes(graph, zip_positions, node_size=50, node_color='red', label='ZIP Codes')

    # Plot route nodes
    route_positions = {n: pos[n] for n in route_nodes}
    nx.draw_networkx_nodes(graph, route_positions, node_size=10, node_color='blue', label='Routes')

    # Plot edges
    nx.draw_networkx_edges(graph, pos, edge_color='gray')

    plt.legend()
    plt.title("ZIP Code Nodes and Evacuation Routes")
    plt.show()
