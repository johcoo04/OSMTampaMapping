import os
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from shapely.geometry import Point

def load_graph_from_pickle(pickle_path):
    """Load the graph from a pickle file."""
    with open(pickle_path, 'rb') as f:
        G = pickle.load(f)
    print(f"Loaded graph from {pickle_path}")
    return G

def filter_closest_terminal_connections(G, n_closest=5):
    """Filter to keep only the n closest terminal connections for each sink."""
    print(f"Filtering to {n_closest} closest terminal connections per sink...")
    
    # Get all sink and terminal nodes
    sink_nodes = [node for node, data in G.nodes(data=True) if data.get("node_type") == "sink"]
    terminal_nodes = [node for node, data in G.nodes(data=True) if data.get("node_type") == "terminal"]
    
    print(f"Found {len(sink_nodes)} sink nodes and {len(terminal_nodes)} terminal nodes")
    
    # Edges to keep
    filtered_edges = []
    
    # Process each sink
    for sink in sink_nodes:
        sink_pos = G.nodes[sink].get("pos")
        sink_zip = G.nodes[sink].get("zip_code", "unknown")
        
        # Calculate distances to all terminals
        terminal_distances = []
        for terminal in terminal_nodes:
            if G.has_edge(sink, terminal):
                terminal_pos = G.nodes[terminal].get("pos")
                dist = Point(sink_pos).distance(Point(terminal_pos))
                terminal_distances.append((dist, terminal))
        
        # Sort by distance and keep the closest n
        terminal_distances.sort()
        closest_terminals = terminal_distances[:n_closest]
        
        # Add these edges to our filtered list
        for _, terminal in closest_terminals:
            filtered_edges.append((sink, terminal))
        
        print(f"Kept {len(closest_terminals)} closest terminals for sink {sink_zip}")
    
    return filtered_edges

def visualize_road_network(G, output_path, n_closest_terminals=5):
    """Visualize the road network with different colors for each road type."""
    print("Visualizing road network...")
    pos = nx.get_node_attributes(G, 'pos')
    
    # Define colors for different road types
    road_type_colors = {
        'P': '#0066CC',       # Primary roads - blue
        'M': '#339933',       # Major roads - green
        'H': '#CC3333',       # Highways - red
        'C': '#9933CC',       # Collector roads - purple
        'R': '#FF9900',       # Ramps - orange
        'CONNECTOR': '#AAAAAA',      # Auto-generated connections - gray
        'CENTROID_CONNECTOR': '#CCFFCC', # Centroid connections - light green
        'SINK_CONNECTOR': '#FFCC99',   # Sink connections - light orange
        'COMPONENT_CONNECTOR': '#FF99FF', # Component connections - light pink
        'SINK_TERMINAL_CONNECTOR': '#FF3333',  # Sink to terminal connections - bright red
        'TERMINAL_CONNECTOR': '#3333FF',  # Terminal to route connections - bright blue
        'UNKNOWN': '#000000'        # Unknown road type - black
    }
    
    # Plot the graph
    plt.figure(figsize=(20, 20))
    
    # Filter to only keep the closest terminal connections
    closest_sink_terminal_edges = filter_closest_terminal_connections(G, n_closest_terminals)
    
    # Draw connectors first (in background)
    connector_types = ['CONNECTOR', 'CENTROID_CONNECTOR', 'COMPONENT_CONNECTOR', 'TERMINAL_CONNECTOR']
    for road_type in connector_types:
        # Get edges of this road type
        edges = [(u, v) for u, v, d in G.edges(data=True) 
                if d.get('road_type') == road_type]
        
        if edges:
            nx.draw_networkx_edges(G, pos, edgelist=edges, 
                                 edge_color=road_type_colors[road_type], 
                                 width=0.7, alpha=0.3)
    
    # Draw actual roads on top with proper colors
    actual_road_types = [t for t in road_type_colors.keys() 
                        if t not in connector_types and t != 'SINK_TERMINAL_CONNECTOR']
    for road_type in actual_road_types:
        # Get edges of this road type
        edges = [(u, v) for u, v, d in G.edges(data=True) 
                if d.get('road_type') == road_type]
        
        if edges:
            nx.draw_networkx_edges(G, pos, edgelist=edges, 
                                 edge_color=road_type_colors[road_type], 
                                 width=1.5, alpha=0.8)
    
    # Draw filtered sink-terminal connections
    if closest_sink_terminal_edges:
        nx.draw_networkx_edges(G, pos, edgelist=closest_sink_terminal_edges, 
                             edge_color=road_type_colors['SINK_TERMINAL_CONNECTOR'], 
                             width=1.2, alpha=0.9)
    
    # Draw centroids as stars
    centroid_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "centroid"]
    if centroid_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=centroid_nodes, node_size=80, 
                             node_color="lime", node_shape='*', alpha=1.0)
    
    # Draw terminal nodes as triangles
    terminal_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "terminal"]
    if terminal_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=terminal_nodes, node_size=80, 
                             node_color="blue", node_shape='^', alpha=1.0)
    
    # Draw sinks as squares
    sink_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "sink"]
    if sink_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=sink_nodes, node_size=80, 
                             node_color="crimson", node_shape='s', alpha=1.0)
    
    # Add labels for sink nodes
    sink_labels = {node: G.nodes[node].get("zip_code", "") for node in sink_nodes}
    nx.draw_networkx_labels(G, pos, labels=sink_labels, font_size=8)
    
    # Add a legend
    legend_items = []
    
    # Road type legend items
    for road_type, color in road_type_colors.items():
        # Only include road types that actually exist in the graph
        if any(d.get('road_type') == road_type for _, _, d in G.edges(data=True)):
            if road_type in connector_types:
                legend_items.append(Line2D([], [], color=color, linewidth=1, alpha=0.3,
                                        label=f'{road_type} (Auto-generated)'))
            elif road_type == 'SINK_TERMINAL_CONNECTOR':
                legend_items.append(Line2D([], [], color=color, linewidth=1.2, alpha=0.9,
                                        label=f'Sink-Terminal Connections (Top {n_closest_terminals})'))
            else:
                road_type_label = {
                    'P': 'Primary',
                    'M': 'Major',
                    'H': 'Highway',
                    'C': 'Collector', 
                    'R': 'Ramp',
                    'UNKNOWN': 'Unknown'
                }.get(road_type, road_type)
                
                legend_items.append(Line2D([], [], color=color, linewidth=2,
                                        label=f'{road_type_label} Roads'))
    
    # Node type legend items
    legend_items.append(Line2D([], [], marker='*', color='w', markerfacecolor='lime', 
                             markersize=10, label='Centroids (Origin)', linestyle='None'))
    legend_items.append(Line2D([], [], marker='^', color='w', markerfacecolor='blue', 
                             markersize=8, label='Terminals', linestyle='None'))
    legend_items.append(Line2D([], [], marker='s', color='w', markerfacecolor='crimson', 
                             markersize=8, label='Sinks (Destination)', linestyle='None'))
    
    plt.legend(handles=legend_items, loc='best')
    plt.title(f"Tampa Evacuation Route Network (Top {n_closest_terminals} Terminal Connections)")
    
    # Remove axes for cleaner look
    plt.axis('off')
    
    # Save the visualization
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Road network visualization saved to {output_path}")
    plt.close()

def main():
    # Define file paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    graph_pickle_path = os.path.join(data_dir, "evacuation_graph.pickle")
    network_viz_path = os.path.join(data_dir, "road_network.png")
    
    # Load the graph
    G = load_graph_from_pickle(graph_pickle_path)
    
    # Visualize just the road network with only top 5 terminal connections
    visualize_road_network(G, network_viz_path, n_closest_terminals=5)

if __name__ == "__main__":
    main()