import os
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from shapely.geometry import Point
import logging

def setup_logging():
    """Simple logging setup"""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
def log_print(message):
    """Log a message"""
    logging.info(message)

def load_graph_from_pickle(pickle_path):
    """Load a NetworkX graph from a pickle file"""
    log_print(f"Loading graph from {pickle_path}")
    with open(pickle_path, 'rb') as f:
        G = pickle.load(f)
    log_print(f"Loaded graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    return G

def visualize_terminals(G, output_path, figsize=(20, 20), dpi=300, label_terminals=True):
    """
    Draw a graph with highlighted and labeled terminal nodes
    
    Args:
        G: NetworkX graph
        output_path: Where to save the visualization
        figsize: Figure size as tuple (width, height) in inches
        dpi: Resolution of the output image
        label_terminals: Whether to add text labels for terminal nodes
    """
    log_print("Creating terminal nodes visualization...")
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Get positions for all nodes
    pos = nx.get_node_attributes(G, 'pos')
    
    # Identify node types for different styles
    terminal_nodes = []
    sink_nodes = []
    centroid_nodes = []
    route_nodes = []
    
    for node, data in G.nodes(data=True):
        if 'pos' not in data:
            continue  # Skip nodes without position
            
        node_type = data.get('node_type', 'unknown')
        if node_type == 'terminal':
            terminal_nodes.append(node)
        elif node_type == 'sink':
            sink_nodes.append(node)
        elif node_type == 'centroid':
            centroid_nodes.append(node)
        elif node_type == 'route':
            route_nodes.append(node)
    
    # Draw routes (base layer, thin gray lines)
    log_print(f"Drawing {len(route_nodes)} route nodes...")
    
    # To avoid drawing all route nodes individually (which could be slow),
    # just draw route edges
    route_edges = []
    for u, v, data in G.edges(data=True):
        if G.nodes.get(u, {}).get('node_type') == 'route' and G.nodes.get(v, {}).get('node_type') == 'route':
            route_edges.append((u, v))
    
    nx.draw_networkx_edges(G, pos, edgelist=route_edges, 
                         edge_color='lightgray', width=0.3, alpha=0.5)
    
    # Draw centroids
    log_print(f"Drawing {len(centroid_nodes)} centroid nodes...")
    nx.draw_networkx_nodes(G, pos, nodelist=centroid_nodes,
                         node_color='blue', node_size=50, alpha=0.7)
    
    # Draw sinks
    log_print(f"Drawing {len(sink_nodes)} sink nodes...")
    nx.draw_networkx_nodes(G, pos, nodelist=sink_nodes,
                         node_color='green', node_size=150, alpha=0.7)
    
    # Draw terminals (larger and more prominent)
    log_print(f"Drawing {len(terminal_nodes)} terminal nodes...")
    nx.draw_networkx_nodes(G, pos, nodelist=terminal_nodes,
                         node_color='red', node_size=200, alpha=1.0)
    
    # Add labels to terminal nodes
    if label_terminals:
        # Create a dictionary for terminal labels with their IDs
        terminal_labels = {}
        for node in terminal_nodes:
            # Extract ID from node name or use ID directly
            terminal_id = node
            if node.startswith('t_'):
                terminal_id = node[2:]  # Remove 't_' prefix
            terminal_labels[node] = terminal_id
        
        nx.draw_networkx_labels(G, pos, labels=terminal_labels,
                              font_size=10, font_color='black', 
                              bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    
    # Add a title and legend
    plt.title("Tampa Evacuation Network with Terminal Nodes")
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15, label='Terminal Nodes'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=12, label='Sink Nodes'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Centroid Nodes'),
        plt.Line2D([0], [0], color='lightgray', lw=2, label='Road Network')
    ]
    
    plt.legend(handles=legend_elements, loc='lower right')
    
    # Turn off axis
    plt.axis('off')
    
    # Save the visualization
    log_print(f"Saving visualization to {output_path}")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    log_print("Visualization complete")
    return output_path

def main():
    """Main entry point"""
    setup_logging()
    
    # Define paths
    pickle_path = "/workspaces/OSMTampaMapping/data/evacuation_graph.pickle"
    output_dir = "/workspaces/OSMTampaMapping/data/results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "terminal_node_visualization.png")
    
    # Load graph and visualize
    G = load_graph_from_pickle(pickle_path)
    visualize_terminals(G, output_path)
    
    log_print(f"Terminal node visualization saved to {output_path}")

if __name__ == "__main__":
    main()