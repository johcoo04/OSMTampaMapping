import networkx as nx
import matplotlib.pyplot as plt
import logging
from loggingFormatterEST import setup_logging  # Import from your custom loggingFormatterEST.py

# Configure logging
logger = setup_logging('visualize.log','visualize')

# Visualize the combined graph with ZIP code nodes and routes.
def visualize_combined_graph(graph):
    pos = nx.get_node_attributes(graph, 'pos')

    # Debugging: Print the pos dictionary
    logger("Positions: %s", pos)

    # Separate ZIP code nodes and route nodes
    zip_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'zip_code']
    route_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'route']

    # Debugging: Print the nodes
    logger("ZIP Nodes: %s", zip_nodes)
    logger("Route Nodes: %s", route_nodes)

    # Ensure all nodes have positions
    for node in graph.nodes():
        if node not in pos:
            logger.warning("Node %s has no position, assigning default position (0, 0)", node)
            pos[node] = (0, 0)  # Assign a default position if missing

    logger.info("All nodes have positions")

    # Plot ZIP code nodes
    zip_positions = {n: pos[n] for n in zip_nodes if n in pos}
    nx.draw_networkx_nodes(graph, zip_positions, node_size=50, node_color='red', label='ZIP Codes')

    # Plot route nodes
    route_positions = {n: pos[n] for n in route_nodes if n in pos}
    nx.draw_networkx_nodes(graph, route_positions, node_size=10, node_color='blue', label='Routes')

    # Plot edges
    nx.draw_networkx_edges(graph, pos, edge_color='gray')

    plt.legend()
    plt.title("ZIP Codes and Evacuation Routes")
    plt.savefig('output.png')
    # plt.show()