import os
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from rtree import index
import pickle
import argparse
import logging
import sys
from ortools.graph.python import min_cost_flow

# Setup logging configuration
def setup_logging(log_file_path):
    """Set up logging to write to both console and file"""
    # Create the directory for the log file if it doesn't exist
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Logging started. Output will be saved to {log_file_path}")

# Replace all print statements with this function
def log_print(message):
    """Log a message to both console and file"""
    logging.info(message)

#Building from scratch
def load_data(routes_path, centroids_path, sink_nodes_path):
    log_print("Loading data...")
    
    # Load evacuation routes
    routes_gdf = gpd.read_file(routes_path).to_crs(epsg=3857)
    log_print(f"Loaded {len(routes_gdf)} evacuation routes")
    
    # Load zip code centroids
    with open(centroids_path, 'r') as f:
        centroids_data = json.load(f)
    centroids = [
        {"geometry": Point(coords["centroid_x"], coords["centroid_y"]), "ZIP_CODE": zip_code}
        for zip_code, coords in centroids_data.items()
    ]
    centroids_gdf = gpd.GeoDataFrame(centroids, crs="EPSG:3857")
    log_print(f"Loaded {len(centroids_gdf)} zip code centroids")
    
    # Load sink nodes
    with open(sink_nodes_path, 'r') as f:
        sink_data = json.load(f)
    sinks = [
        {"geometry": Point(coords["centroid_x"], coords["centroid_y"]), "ZIP_CODE": zip_code}
        for zip_code, coords in sink_data.items()
    ]
    sinks_gdf = gpd.GeoDataFrame(sinks, crs="EPSG:3857")
    log_print(f"Loaded {len(sinks_gdf)} sink nodes")
    
    return routes_gdf, centroids_gdf, sinks_gdf

def create_connected_graph(routes_gdf, centroids_gdf, sinks_gdf, connection_threshold=100):
    """Create a connected graph with centroids, routes, and sinks."""
    log_print("Creating connected graph...")
    start_time = time.time()
    
    # Initialize graph
    G = nx.Graph()
    
    # First pass: Add all route points and connect consecutive points in each route
    log_print("Adding route segments to graph...")
    for idx, route in routes_gdf.iterrows():
        coords = list(route.geometry.coords)
        # Get the road type from properties
        road_type = route.get('TYPE', 'UNKNOWN')
        
        for i in range(len(coords) - 1):
            start = coords[i]
            end = coords[i + 1]
            start_node = f"r{idx}_{i}"
            end_node = f"r{idx}_{i+1}"
            
            # Add nodes and edges for each segment, including road type
            G.add_node(start_node, pos=start, node_type="route", road_type=road_type)
            G.add_node(end_node, pos=end, node_type="route", road_type=road_type)
            G.add_edge(start_node, end_node, weight=Point(start).distance(Point(end)), road_type=road_type)
    
    log_print(f"Added {len(routes_gdf)} routes with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    # Second pass: Build spatial index for all route nodes
    log_print("Building spatial index for route nodes...")
    route_nodes = [(node, data["pos"]) for node, data in G.nodes(data=True) if data["node_type"] == "route"]
    
    # Create spatial index
    spatial_idx = index.Index()
    for i, (node, pos) in enumerate(route_nodes):
        # Index format: (id, (left, bottom, right, top))
        spatial_idx.insert(i, (pos[0], pos[1], pos[0], pos[1]))
    
    # Third pass: Connect nearby route nodes
    log_print("Connecting nearby route nodes...")
    connections_made = 0
    
    for i, (node1, pos1) in enumerate(route_nodes):
        # Query the spatial index for nearby nodes
        nearby = list(spatial_idx.intersection((
            pos1[0] - connection_threshold, 
            pos1[1] - connection_threshold,
            pos1[0] + connection_threshold, 
            pos1[1] + connection_threshold
        )))
        
        for j in nearby:
            node2, pos2 = route_nodes[j]
            if node1 != node2:  # Don't connect to self
                dist = Point(pos1).distance(Point(pos2))
                if dist <= connection_threshold and not G.has_edge(node1, node2):
                    # When connecting nodes from different routes, mark as "connector"
                    G.add_edge(node1, node2, weight=dist, road_type="CONNECTOR")
                    connections_made += 1
    
    log_print(f"Made {connections_made} new connections between nearby route nodes")
    
    # Fourth pass: Check connectivity of route network
    log_print("Checking route network connectivity...")
    route_subgraph = G.subgraph([n for n, d in G.nodes(data=True) if d["node_type"] == "route"])
    
    if nx.is_connected(route_subgraph):
        log_print("Route network is fully connected!")
    else:
        components = list(nx.connected_components(route_subgraph))
        log_print(f"Warning: Route network has {len(components)} disconnected components")
        largest = max(components, key=len)
        log_print(f"Largest component has {len(largest)} nodes ({len(largest)/len(route_subgraph.nodes())*100:.1f}% of route nodes)")
        
        # If connectivity is very poor, we could try increasing the threshold and reconnecting
        if len(largest)/len(route_subgraph.nodes()) < 0.9 and connection_threshold < 500:
            log_print(f"Poor connectivity detected. Attempting to connect major components with increased threshold...")
            # Find the two largest components
            components_sorted = sorted(components, key=len, reverse=True)
            if len(components_sorted) >= 2:
                comp1 = list(components_sorted[0])
                for comp_idx in range(1, min(5, len(components_sorted))):
                    comp2 = list(components_sorted[comp_idx])
                    # Find closest pair of nodes between components
                    min_dist = float('inf')
                    closest_pair = None
                    for n1 in comp1:
                        for n2 in comp2:
                            pos1 = G.nodes[n1]['pos']
                            pos2 = G.nodes[n2]['pos']
                            dist = Point(pos1).distance(Point(pos2))
                            if dist < min_dist:
                                min_dist = dist
                                closest_pair = (n1, n2)
                    
                    if closest_pair:
                        G.add_edge(closest_pair[0], closest_pair[1], weight=min_dist, road_type="CONNECTOR")
                        log_print(f"Connected components by adding edge with distance {min_dist:.2f} meters")
                        comp1.extend(comp2)  # Merge components
    
    # Fifth pass: Add centroids and connect to multiple nearest route nodes
    log_print("Adding and connecting centroids...")
    for cent_idx, centroid in centroids_gdf.iterrows():
        centroid_id = f"c{cent_idx}"
        centroid_point = centroid.geometry
        centroid_pos = (centroid_point.x, centroid_point.y)
        
        # Add centroid to graph
        G.add_node(centroid_id, pos=centroid_pos, node_type="centroid", zip_code=centroid["ZIP_CODE"])
        
        # Find nearby route nodes using spatial index
        nearby = list(spatial_idx.intersection((
            centroid_pos[0] - 5000,  # Use a larger search radius initially
            centroid_pos[1] - 5000,
            centroid_pos[0] + 5000,
            centroid_pos[1] + 5000
        )))
        
        # Calculate distances to all nearby nodes
        distances = []
        for j in nearby:
            route_node, route_pos = route_nodes[j]
            dist = Point(centroid_pos).distance(Point(route_pos))
            distances.append((dist, route_node))
        
        # Sort by distance and connect to the closest 3 (or fewer if not enough)
        distances.sort()
        connection_count = min(3, len(distances))
        
        if connection_count > 0:
            for dist, node in distances[:connection_count]:
                G.add_edge(centroid_id, node, weight=dist, road_type="CENTROID_CONNECTOR")
            log_print(f"Connected centroid {centroid['ZIP_CODE']} to {connection_count} route nodes")
        else:
            log_print(f"Warning: Could not find nearby route nodes for centroid {centroid['ZIP_CODE']}")
    
    # Sixth pass: Add sink nodes and connect to multiple nearest route nodes
    log_print("Adding and connecting sink nodes...")
    for sink_idx, sink in sinks_gdf.iterrows():
        sink_id = f"s{sink_idx}"
        sink_point = sink.geometry
        sink_pos = (sink_point.x, sink_point.y)
        
        # Add sink to graph
        G.add_node(sink_id, pos=sink_pos, node_type="sink", zip_code=sink["ZIP_CODE"])
        
        # Find nearby route nodes using spatial index
        nearby = list(spatial_idx.intersection((
            sink_pos[0] - 5000,  # Use a larger search radius initially
            sink_pos[1] - 5000,
            sink_pos[0] + 5000,
            sink_pos[1] + 5000
        )))
        
        # Calculate distances to all nearby nodes
        distances = []
        for j in nearby:
            route_node, route_pos = route_nodes[j]
            dist = Point(sink_pos).distance(Point(route_pos))
            distances.append((dist, route_node))
        
        # Sort by distance and connect to the closest 3 (or fewer if not enough)
        distances.sort()
        connection_count = min(3, len(distances))
        
        if connection_count > 0:
            for dist, node in distances[:connection_count]:
                G.add_edge(sink_id, node, weight=dist, road_type="SINK_CONNECTOR")
            log_print(f"Connected sink {sink['ZIP_CODE']} to {connection_count} route nodes")
        else:
            log_print(f"Warning: Could not find nearby route nodes for sink {sink['ZIP_CODE']}")
    
    # Final check: Is the entire graph connected?
    log_print("Checking overall graph connectivity...")
    if nx.is_connected(G):
        log_print("Success! Graph is fully connected.")
    else:
        components = list(nx.connected_components(G))
        log_print(f"Warning: Graph has {len(components)} disconnected components")
        largest = max(components, key=len)
        log_print(f"Largest component has {len(largest)} nodes ({len(largest)/len(G.nodes())*100:.1f}% of all nodes)")
        
        # Report how many centroid and sink nodes are in the main component
        centroids_in_main = sum(1 for node in largest if G.nodes[node].get("node_type") == "centroid")
        sinks_in_main = sum(1 for node in largest if G.nodes[node].get("node_type") == "sink")
        log_print(f"Main component contains {centroids_in_main}/{len(centroids_gdf)} centroids and {sinks_in_main}/{len(sinks_gdf)} sinks")
        
        # Try to find and connect the remaining components
        if len(components) > 1:
            log_print("Attempting to connect remaining components...")
            components_sorted = sorted(components, key=len, reverse=True)
            main_comp = components_sorted[0]
            
            for comp_idx in range(1, len(components_sorted)):
                other_comp = components_sorted[comp_idx]
                # Find closest pair of nodes between components
                min_dist = float('inf')
                closest_pair = None
                
                for n1 in main_comp:
                    for n2 in other_comp:
                        if 'pos' in G.nodes[n1] and 'pos' in G.nodes[n2]:
                            pos1 = G.nodes[n1]['pos']
                            pos2 = G.nodes[n2]['pos']
                            dist = Point(pos1).distance(Point(pos2))
                            if dist < min_dist:
                                min_dist = dist
                                closest_pair = (n1, n2)
                
                if closest_pair:
                    G.add_edge(closest_pair[0], closest_pair[1], weight=min_dist, road_type="COMPONENT_CONNECTOR")
                    log_print(f"Connected component {comp_idx} to main component with edge of distance {min_dist:.2f} meters")
                    main_comp = main_comp.union(other_comp)  # Update main component
    
    # Final connectivity check after attempted fixes
    if not nx.is_connected(G):
        log_print("Warning: Graph remains disconnected after attempted fixes")
        
        # Identify which centroids and sinks are disconnected
        components = list(nx.connected_components(G))
        main_comp = max(components, key=len)
        
        disconnected_centroids = [G.nodes[node]["zip_code"] for node in G.nodes() 
                                if G.nodes[node].get("node_type") == "centroid" and node not in main_comp]
        disconnected_sinks = [G.nodes[node]["zip_code"] for node in G.nodes() 
                            if G.nodes[node].get("node_type") == "sink" and node not in main_comp]
        
        if disconnected_centroids:
            log_print(f"Disconnected centroids: {', '.join(disconnected_centroids)}")
        if disconnected_sinks:
            log_print(f"Disconnected sinks: {', '.join(disconnected_sinks)}")
    
    total_time = time.time() - start_time
    log_print(f"Graph creation completed in {total_time:.2f} seconds")
    return G

def visualize_path_with_road_types(G, path, output_path, centroid_zip=""):
    """Visualize the shortest path with color-coded road types."""
    log_print(f"Visualizing shortest path for {centroid_zip}...")
    pos = nx.get_node_attributes(G, 'pos')
    
    # Define colors for different road types
    road_type_colors = {
        'P': 'blue',              # Primary roads
        'M': 'green',             # Major roads
        'H': 'red',               # Highways
        'C': 'purple',            # Collector roads
        'CONNECTOR': 'gray',      # Auto-generated connections
        'CENTROID_CONNECTOR': 'lime', # Centroid connections (usually hidden)
        'SINK_CONNECTOR': 'orange',   # Sink connections (usually hidden)
        'COMPONENT_CONNECTOR': 'magenta', # Component connections
        'SINK_TERMINAL_CONNECTOR': 'orange', # Sink to terminal connections
        'TERMINAL_CONNECTOR': 'cyan',  # Terminal to route connections
        'UNKNOWN': 'black'        # Unknown road type
    }
    
    # Plot the graph
    plt.figure(figsize=(15, 15))
    
    # Draw route edges by road type
    for road_type, color in road_type_colors.items():
        # Get edges of this road type
        edges = [(u, v) for u, v, d in G.edges(data=True) 
                if d.get('road_type') == road_type]
        
        # Skip if no edges of this type
        if not edges:
            continue
        
        # Adjust alpha based on type
        alpha = 0.15 if road_type in ['CONNECTOR', 'CENTROID_CONNECTOR', 'SINK_CONNECTOR', 'SINK_TERMINAL_CONNECTOR'] else 0.5
        width = 0.7 if road_type in ['CONNECTOR', 'CENTROID_CONNECTOR', 'SINK_CONNECTOR', 'SINK_TERMINAL_CONNECTOR'] else 1.5
        
        # Draw this road type with appropriate color
        nx.draw_networkx_edges(G, pos, edgelist=edges, 
                              edge_color=color, width=width, alpha=alpha)
    
    # Draw centroid nodes in bright green - FIX to avoid warning
    centroid_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "centroid"]
    if centroid_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=centroid_nodes, node_size=80, 
                             node_color="lime", alpha=1.0)
    
    # Draw sink nodes in bright red - FIX to avoid warning
    sink_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "sink"]
    if sink_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=sink_nodes, node_size=80, 
                             node_color="crimson", alpha=1.0)
    
    # Draw terminal nodes in bright blue
    terminal_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "terminal"]
    if terminal_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=terminal_nodes, node_size=80, 
                             node_color="blue", alpha=1.0)
    
    # Highlight the current centroid if provided
    if centroid_zip:
        current_centroid = [n for n, d in G.nodes(data=True) 
                          if d.get("node_type") == "centroid" and d.get("zip_code") == centroid_zip]
        if current_centroid:
            nx.draw_networkx_nodes(G, pos, nodelist=current_centroid, node_size=150, 
                                node_color="yellow", alpha=1.0, linewidths=2, edgecolors="black")
    
    # If path exists, highlight it
    if path:
        # For path nodes - FIX to avoid warning
        if path:
            nx.draw_networkx_nodes(G, pos, nodelist=path, node_size=30, 
                                node_color="yellow", alpha=0.8)
        
        # For path edges, color by road type
        path_edges = list(zip(path, path[1:]))
        for u, v in path_edges:
            if G.has_edge(u, v):
                road_type = G.edges[u, v].get('road_type', 'UNKNOWN')
                color = road_type_colors.get(road_type, 'black')
                # For artificial connections in the path, use yellow highlight
                if road_type in ['CONNECTOR', 'CENTROID_CONNECTOR', 'SINK_CONNECTOR', 'COMPONENT_CONNECTOR', 'SINK_TERMINAL_CONNECTOR']:
                    color = 'yellow'
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                                      edge_color=color, width=3, alpha=1.0)
    
    # Add a legend
    legend_items = []
    
    # Road types for legend
    for road_type, color in road_type_colors.items():
        # Only show in legend if there are edges of this type
        if any(d.get('road_type') == road_type for _, _, d in G.edges(data=True)):
            if road_type in ['CONNECTOR', 'CENTROID_CONNECTOR', 'SINK_CONNECTOR', 'COMPONENT_CONNECTOR', 'SINK_TERMINAL_CONNECTOR']:
                legend_items.append(plt.Line2D([], [], color=color, linewidth=1, alpha=0.3,
                        label=f'{road_type} (Auto-generated)'))
            else:
                legend_items.append(plt.Line2D([], [], color=color, linewidth=2, 
                        label=f'{road_type} Roads'))
    
    # Node types for legend
    legend_items.append(plt.Line2D([], [], marker='o', color='w', markerfacecolor='lime', 
                       markersize=8, label='Centroids', linestyle='None'))
    legend_items.append(plt.Line2D([], [], marker='o', color='w', markerfacecolor='crimson', 
                       markersize=8, label='Sinks', linestyle='None'))
    legend_items.append(plt.Line2D([], [], marker='o', color='w', markerfacecolor='blue', 
                       markersize=8, label='Terminals', linestyle='None'))
    
    if centroid_zip:
        legend_items.append(plt.Line2D([], [], marker='o', color='w', markerfacecolor='yellow', 
                           markersize=10, markeredgecolor='black', label=f'Centroid {centroid_zip}', 
                           linestyle='None'))
    
    if path:
        legend_items.append(plt.Line2D([], [], marker='o', color='w', markerfacecolor='yellow', 
                           markersize=6, label='Path Nodes', linestyle='None'))
        legend_items.append(plt.Line2D([], [], color='yellow', linewidth=3, label='Connection in Path'))
    
    plt.legend(handles=legend_items, loc='best')
    
    # Add title with path information if available
    if path and centroid_zip:
        sink_zip = ""
        for node in path:
            if G.nodes[node].get("node_type") == "sink":
                sink_zip = G.nodes[node].get("zip_code", "")
                break
        
        title = f"Evacuation Path: {centroid_zip} to {sink_zip}"
        if sink_zip:
            path_length = nx.shortest_path_length(G, path[0], path[-1], weight="weight")
            title += f" ({path_length/1000:.2f} km)"
    else:
        title = "Evacuation Route Network by Road Type"
    
    plt.title(title)
    plt.savefig(output_path, dpi=300)
    log_print(f"Visualization saved to {output_path}")
    plt.close()

def connect_sinks_to_terminals(G, terminal_nodes_path, sink_nodes_path):
    
    log_print("Connecting sink nodes to terminal nodes...")
    
    # Load terminal nodes from JSON
    with open(terminal_nodes_path, 'r') as f:
        terminal_data = json.load(f)
    
    # Load sink nodes from JSON
    with open(sink_nodes_path, 'r') as f:
        sink_data = json.load(f)
    
    log_print(f"Loaded {len(terminal_data)} terminal nodes and {len(sink_data)} sink nodes from JSON")
    
    # First, ensure all terminal nodes are in the graph
    terminal_node_ids = []
    for terminal_id, coords in terminal_data.items():
        node_id = f"t_{terminal_id}"
        terminal_pos = (coords["x"], coords["y"])
        
        # Add terminal node if not already in graph
        if node_id not in G:
            G.add_node(node_id, pos=terminal_pos, node_type="terminal", road_type="TERMINAL")
            log_print(f"Added terminal node {terminal_id} to graph")
        else:
            # Update position if node exists
            G.nodes[node_id]["pos"] = terminal_pos
            G.nodes[node_id]["node_type"] = "terminal"
            G.nodes[node_id]["road_type"] = "TERMINAL"
        
        terminal_node_ids.append(node_id)
    
    # Ensure terminals are connected to the route network
    route_nodes = [node for node, data in G.nodes(data=True) if data.get("node_type") == "route"]
    
    # Build spatial index for route nodes
    spatial_idx = index.Index()
    for i, node in enumerate(route_nodes):
        pos = G.nodes[node]["pos"]
        # Index format: (id, (left, bottom, right, top))
        spatial_idx.insert(i, (pos[0], pos[1], pos[0], pos[1]))
    
    # Connect each terminal to nearest route nodes
    for terminal_id in terminal_node_ids:
        terminal_pos = G.nodes[terminal_id]["pos"]
        
        # Skip if terminal already has route connections
        has_route_connection = any(G.nodes[neighbor].get("node_type") == "route" 
                                  for neighbor in G.neighbors(terminal_id))
        
        if not has_route_connection:
            # Find nearby route nodes
            nearby = list(spatial_idx.intersection((
                terminal_pos[0] - 1000,  # 1km search radius
                terminal_pos[1] - 1000,
                terminal_pos[0] + 1000,
                terminal_pos[1] + 1000
            )))
            
            if nearby:
                # Connect to closest route node
                closest_idx = nearby[0]
                closest_node = route_nodes[closest_idx]
                closest_pos = G.nodes[closest_node]["pos"]
                dist = Point(terminal_pos).distance(Point(closest_pos))
                
                G.add_edge(terminal_id, closest_node, weight=dist, road_type="TERMINAL_CONNECTOR")
                log_print(f"Connected terminal {terminal_id} to route node at distance {dist/1000:.2f} km")
            else:
                log_print(f"Warning: Could not find nearby route nodes for terminal {terminal_id}")
    
    # Find all sink nodes in the graph
    sink_node_ids = []
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "sink":
            sink_node_ids.append(node)
    
    # Remove all existing connections from sink nodes
    for sink_id in sink_node_ids:
        sink_zip = G.nodes[sink_id].get("zip_code", "unknown")
        neighbors = list(G.neighbors(sink_id))
        
        for neighbor in neighbors:
            G.remove_edge(sink_id, neighbor)
        
        log_print(f"Removed {len(neighbors)} existing connections from sink {sink_zip}")
    
    # Connect each sink to each terminal
    total_connections = 0
    for sink_id in sink_node_ids:
        sink_zip = G.nodes[sink_id].get("zip_code", "unknown")
        sink_pos = G.nodes[sink_id]["pos"]
        
        for terminal_id in terminal_node_ids:
            terminal_pos = G.nodes[terminal_id]["pos"]
            dist = Point(sink_pos).distance(Point(terminal_pos))
            
            G.add_edge(sink_id, terminal_id, weight=dist, road_type="SINK_TERMINAL_CONNECTOR")
            total_connections += 1
        
        log_print(f"Connected sink {sink_zip} to all {len(terminal_node_ids)} terminal nodes")
    
    log_print(f"Created {total_connections} connections between sinks and terminals")
    return G

def save_graph_to_pickle(G, pickle_path):
    """Save the graph to a pickle file."""
    with open(pickle_path, 'wb') as f:
        pickle.dump(G, f)
    log_print(f"Saved graph to {pickle_path}")

def load_graph_from_pickle(pickle_path):
    """Load the graph from a pickle file."""
    with open(pickle_path, 'rb') as f:
        G = pickle.load(f)
    log_print(f"Loaded graph from {pickle_path}")
    return G

#Building from pickle
def process_all_centroids(G, centroids_gdf, output_dir):
    """Process all centroids, find paths to the top 3 closest sinks using A*."""
    log_print("Computing shortest paths to multiple sinks using A* algorithm...")
    start_time = time.time()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all sink and centroid nodes
    sink_nodes = [node for node, data in G.nodes(data=True) if data.get("node_type") == "sink"]
    centroid_nodes = {}
    
    # Map ZIP codes to centroid node IDs
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "centroid":
            centroid_nodes[data.get("zip_code")] = node
    
    # Define a heuristic function for A* (Euclidean distance)
    def heuristic(n1, n2):
        pos1 = G.nodes[n1]['pos']
        pos2 = G.nodes[n2]['pos']
        return Point(pos1).distance(Point(pos2))
    
    # Process each centroid
    results = []
    for _, centroid in centroids_gdf.iterrows():
        zip_code = centroid["ZIP_CODE"]
        log_print(f"Processing centroid {zip_code}...")
        
        # Get the centroid node
        centroid_node = centroid_nodes.get(zip_code)
        if not centroid_node:
            log_print(f"Warning: Centroid {zip_code} not found in the graph!")
            continue
        
        # Find paths to all sinks using A*
        sink_distances = []
        for sink_node in sink_nodes:
            try:
                # Get both the path and its length
                path = nx.astar_path(G, centroid_node, sink_node, 
                                    heuristic=heuristic, weight="weight")
                distance = sum(G[u][v]['weight'] for u, v in zip(path, path[1:]))
                
                sink_zip = G.nodes[sink_node].get("zip_code")
                
                # Count road types in the path
                road_types_in_path = {}
                for u, v in zip(path, path[1:]):
                    road_type = G.edges[u, v].get('road_type', 'UNKNOWN')
                    road_types_in_path[road_type] = road_types_in_path.get(road_type, 0) + 1
                
                sink_distances.append({
                    "sink": sink_zip,
                    "sink_node": sink_node,
                    "distance_meters": distance,
                    "distance_km": distance/1000,
                    "path_nodes": len(path),
                    "path": path,
                    "road_types": road_types_in_path
                })
            except nx.NetworkXNoPath:
                continue
        
        if not sink_distances:
            log_print(f"No path found from centroid {zip_code} to any sink!")
            results.append({"centroid": zip_code, "paths": []})
            continue
            
        # Sort by distance and get top 3 sinks (or fewer if not available)
        sink_distances.sort(key=lambda x: x["distance_meters"])
        top_paths = sink_distances[:min(3, len(sink_distances))]
        
        if top_paths:
            # Comment out visualization to save time
            # shortest_path_data = top_paths[0]
            # output_path = os.path.join(output_dir, f"path_{zip_code}_to_{shortest_path_data['sink']}.png")
            # visualize_path_with_road_types(G, shortest_path_data["path"], output_path, zip_code)
            
            # Add result to collection
            result = {
                "centroid": zip_code,
                "paths": top_paths  # Top 3 paths to different sinks
            }
            results.append(result)
            
            log_print(f"Found {len(top_paths)} paths from centroid {zip_code} to different sinks")
            for i, path_data in enumerate(top_paths):
                log_print(f"  Path {i+1}: to sink {path_data['sink']}, distance: {path_data['distance_km']:.2f} km")
        else:
            log_print(f"No path found from centroid {zip_code} to any sink!")
            results.append({
                "centroid": zip_code,
                "paths": []
            })
    
    # Create a summary file
    summary_path = os.path.join(output_dir, "evacuation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Evacuation Path Summary\n")
        f.write("======================\n\n")
        
        for result in results:
            f.write(f"Centroid: {result['centroid']}\n")
            
            if result['paths']:
                # First closest sink (most detailed)
                path1 = result['paths'][0]
                f.write(f"Closest Sink: {path1['sink']}\n")
                f.write(f"Distance: {path1['distance_km']:.2f} km ({path1['distance_meters']:.2f} m)\n")
                f.write(f"Path Nodes: {path1['path_nodes']}\n")
                
                if path1['road_types']:
                    f.write("Road Types Used:\n")
                    for road_type, count in path1['road_types'].items():
                        f.write(f"  {road_type}: {count} segments\n")
                
                # Second closest sink (if available)
                if len(result['paths']) > 1:
                    path2 = result['paths'][1]
                    f.write(f"\n2nd Closest Sink: {path2['sink']}\n")
                    f.write(f"2nd Closest Distance: {path2['distance_km']:.2f} km ({path2['distance_meters']:.2f} m)\n")
                
                # Third closest sink (if available)
                if len(result['paths']) > 2:
                    path3 = result['paths'][2]
                    f.write(f"\n3rd Closest Sink: {path3['sink']}\n")
                    f.write(f"3rd Closest Distance: {path3['distance_km']:.2f} km ({path3['distance_meters']:.2f} m)\n")
            else:
                f.write("No path found to any sink.\n")
            
            f.write("\n" + "-"*40 + "\n\n")
    
    total_time = time.time() - start_time
    log_print(f"Pathfinding completed in {total_time:.2f} seconds")
    log_print(f"Summary saved to {summary_path}")
    return results

def assign_road_capacities_and_population(G, population_data_path, sink_nodes_path):
    log_print("Assigning road capacities and population data...")
    
    # Define road properties (capacity in people/hour only - doubled from original)
    road_properties = {
        'H': {'capacity': 40000},  # Highway
        'P': {'capacity': 16000},  # Primary road
        'M': {'capacity': 14000},  # Major road
        'C': {'capacity': 20000},  # Collector road
        'R': {'capacity': 6000},   # Ramp
        'CONNECTOR': {'capacity': 40000},  # Artificial connectors
        'CENTROID_CONNECTOR': {'capacity': 40000},
        'SINK_CONNECTOR': {'capacity': 40000},
        'COMPONENT_CONNECTOR': {'capacity': 40000},
        'SINK_TERMINAL_CONNECTOR': {'capacity': 40000},
        'TERMINAL_CONNECTOR': {'capacity': 40000},
        'UNKNOWN': {'capacity': 40000}
    }
    
    # Define default properties to use when a road type isn't found
    default_properties = {'capacity': 1000}
    
    # Load population data - array format
    with open(population_data_path, 'r') as f:
        population_array = json.load(f)
    
    # Convert array to dictionary with zip codes as keys
    population_data = {}
    for item in population_array:
        zip_code = item.get("zip")
        if zip_code:
            population_data[zip_code] = {"population": item.get("population", 0)}
    
    log_print(f"Loaded population data for {len(population_data)} zip codes")
    
    # Load sink node capacities
    with open(sink_nodes_path, 'r') as f:
        sink_data = json.load(f)
    
    log_print(f"Loaded capacity data for {len(sink_data)} sink nodes")
    
    # Assign capacity to all edges based on road type
    edges_updated = 0
    for u, v, data in G.edges(data=True):
        road_type = data.get('road_type', 'UNKNOWN')
        # Use first character if it's a single-character road type
        road_key = road_type[0] if len(road_type) == 1 else road_type
        
        # Get properties with a safe fallback
        props = road_properties.get(road_key, default_properties)
        
        # Assign capacity
        G[u][v]['capacity'] = props['capacity']
        
        edges_updated += 1
    
    log_print(f"Updated {edges_updated} edges with capacity attributes")
    
    # Assign population as 'demand' to centroid nodes (negative for sources)
    centroids_updated = 0
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "centroid":
            zip_code = data.get("zip_code")
            if zip_code in population_data:
                population = population_data[zip_code].get("population", 0)
                # Set as negative for sources in min cost flow
                G.nodes[node]['demand'] = -population
                centroids_updated += 1
            else:
                log_print(f"Warning: No population data found for zip code {zip_code}")
                G.nodes[node]['demand'] = 0
    
    log_print(f"Updated {centroids_updated} centroids with population demand")
    
    # Assign capacity values to sink nodes from the sink_nodes.json file
    sinks_updated = 0
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "sink":
            zip_code = data.get("zip_code")
            if zip_code in sink_data:
                # Set the capacity from the sink_nodes.json file - increase by 2x
                capacity = sink_data[zip_code].get("population", 1999999) * 2
                G.nodes[node]['demand'] = capacity
                G.nodes[node]['capacity'] = capacity
                sinks_updated += 1
            else:
                log_print(f"Warning: No capacity data found for sink with zip code {zip_code}")
                # Default fallback capacity
                G.nodes[node]['demand'] = 3999998
                G.nodes[node]['capacity'] = 3999998
    
    log_print(f"Updated {sinks_updated} sink nodes with capacity values from sink_nodes.json")
    
    return G

def analyze_evacuation_flow_ortools(G, output_dir, phases=4, debug=False):

    log_print(f"Analyzing evacuation flow using {phases}-phase evacuation model with OR-Tools...")
    start_time = time.time()
    
    # Identify source (centroid) nodes with population data
    source_nodes = []
    sink_nodes = []
    
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "centroid" and data.get("demand", 0) < 0:
            population = -data.get("demand", 0)
            zip_code = data.get("zip_code", "unknown")
            source_nodes.append((node, population, zip_code))
        elif data.get("node_type") == "sink":
            sink_nodes.append(node)
    
    # Sort source nodes by population in descending order
    source_nodes.sort(key=lambda x: x[1], reverse=True)
    total_population = sum(pop for _, pop, _ in source_nodes)
    
    log_print(f"Planning evacuation for {total_population:,} people from {len(source_nodes)} zip codes")
    log_print(f"Dividing into {phases} evacuation waves")
    
    # Divide sources into phases
    sources_per_phase = len(source_nodes) // phases
    remainder = len(source_nodes) % phases
    
    phase_sources = []
    start_idx = 0
    
    for i in range(phases):
        extra = 1 if i < remainder else 0
        end_idx = start_idx + sources_per_phase + extra
        phase_sources.append(source_nodes[start_idx:end_idx])
        start_idx = end_idx
    
    # Process each phase
    phase_results = []
    cumulative_time = 0
    cumulative_output = []
    
    for phase_idx, sources in enumerate(phase_sources):
        phase_start_time = time.time()
        phase_num = phase_idx + 1
        log_print(f"\nProcessing Phase {phase_num} with {len(sources)} source nodes")
        
        # Add detailed flow information for each centroid
        if debug:
            log_print("\nCentroid evacuation capacity breakdown:")
            for node, pop, zip_code in sources:
                outgoing_capacity = sum(G[node][neighbor].get('capacity', 0) for neighbor in G.neighbors(node))
                log_print(f"  ZIP {zip_code}: {pop:,} people / {outgoing_capacity:,} capacity per hour")
        
        # Create phase-specific graph copy
        F = G.copy()
        
        # Reset all demands
        for node in F.nodes():
            if 'demand' in F.nodes[node]:
                F.nodes[node]['demand'] = 0
        
        # Set demand only for current phase sources
        phase_pop = 0
        for node, pop, zip_code in sources:
            F.nodes[node]['demand'] = -pop
            phase_pop += pop
            log_print(f"  Phase {phase_num} source: {zip_code} with population {pop:,}")
        
        # Create a super sink with adjusted capacity
        super_sink = f"super_sink_phase_{phase_num}"
        F.add_node(super_sink, demand=phase_pop)
        
        # Connect sinks to super sink with proper capacity
        for sink in sink_nodes:
            # Use 10x phase population for sink capacity
            sink_capacity = phase_pop * 10
            F.add_edge(sink, super_sink, capacity=sink_capacity, weight=1)
            log_print(f"  Connecting sink {sink} to super sink with capacity {sink_capacity:,}")
        
        log_print(f"Phase {phase_num} total population: {phase_pop:,}")
        
        try:
            # Create the min cost flow solver
            mcf = min_cost_flow.SimpleMinCostFlow()
            
            # Create node maps for OR-Tools
            node_map = {}
            reverse_map = {}
            
            # Map nodes to integers for OR-Tools
            for i, node in enumerate(F.nodes()):
                node_map[node] = i
                reverse_map[i] = node
            
            # Add each node with demand
            for node, data in F.nodes(data=True):
                demand = data.get('demand', 0)
                if demand != 0:
                    mcf.set_node_supply(node_map[node], int(demand))
            
            # Add each edge as an arc - use existing capacities
            for u, v, data in F.edges(data=True):
                # Get the existing capacity from the graph (previously assigned)
                # This respects the capacities from assign_road_capacities_and_population
                base_capacity = int(data.get('capacity', 1000))
                
                # Only modify capacities for specific connection types
                if F.nodes[u].get('node_type') == 'centroid' or F.nodes[v].get('node_type') == 'centroid':
                    # Ensure centroids can evacuate their entire population
                    capacity = max(base_capacity, phase_pop)
                elif u == super_sink or v == super_sink:
                    # Super sink connections need high capacity
                    capacity = phase_pop * 10
                else:
                    # For all other edges, use the capacity assigned in 
                    # assign_road_capacities_and_population
                    capacity = base_capacity
                
                # Use unit weights for simplicity
                weight = 1
                
                # Add the arc
                mcf.add_arc_with_capacity_and_unit_cost(
                    node_map[u], node_map[v], capacity, weight)
                
                # For undirected graphs, add reverse edge
                if not F.is_directed():
                    mcf.add_arc_with_capacity_and_unit_cost(
                        node_map[v], node_map[u], capacity, weight)
            
            # Find the min cost flow
            log_print("  Starting OR-Tools MinCostFlow computation...")
            computation_start = time.time()
            status = mcf.solve()
            computation_time = time.time() - computation_start
            log_print(f"  OR-Tools computation completed in {computation_time:.2f} seconds")
            
            if status == mcf.OPTIMAL:
                log_print("  Optimal solution found!")
                
                # Extract flow results
                max_flow = 0
                bottleneck_edge = None
                congested_edges = []
                total_flow = 0
                used_flow_paths = 0
                
                for i in range(mcf.num_arcs()):
                    if mcf.flow(i) > 0:
                        # Get arc endpoints
                        tail = mcf.tail(i)
                        head = mcf.head(i)
                        flow = mcf.flow(i)
                        capacity = mcf.capacity(i)
                        
                        # Map back to original node IDs
                        orig_u = reverse_map[tail]
                        orig_v = reverse_map[head]
                        
                        # Skip super sink connections for bottleneck analysis
                        if orig_v != super_sink and orig_u != super_sink:
                            used_flow_paths += 1
                            total_flow += flow
                            
                            if flow > max_flow:
                                max_flow = flow
                                bottleneck_edge = (orig_u, orig_v)
                            
                            # Check for congestion
                            if capacity > 0 and flow / capacity > 0.8:
                                road_type = 'UNKNOWN'
                                if F.has_edge(orig_u, orig_v):
                                    road_type = F[orig_u][orig_v].get('road_type', 'UNKNOWN')
                                    
                                congested_edges.append((orig_u, orig_v, flow, capacity, flow/capacity, road_type))
                
                # Calculate evacuation time for this phase
                phase_time = phase_pop / max(1, max_flow)
                
                if debug:
                    log_print(f"  Total evacuation flow: {total_flow:,} units")
                    log_print(f"  Used {used_flow_paths} flow paths")
                
                if bottleneck_edge:
                    u, v = bottleneck_edge
                    log_print(f"  Bottleneck: {u} → {v} with flow {max_flow:,} people/hour")
                    if F.has_edge(u, v):
                        road_type = F[u][v].get('road_type', 'UNKNOWN')
                        log_print(f"  Bottleneck road type: {road_type}")
                    
                    log_print(f"  Found {len(congested_edges)} congested road segments (>80% capacity)")
                    if congested_edges:
                        for i, (u, v, flow, capacity, ratio, road_type) in enumerate(
                                sorted(congested_edges, key=lambda x: x[4], reverse=True)[:5]):
                            log_print(f"    {i+1}. {u} → {v} ({road_type}): {flow:,}/{capacity:,} ({ratio*100:.1f}%)")
                
                # Store results
                phase_results.append({
                    'phase': phase_num,
                    'population': phase_pop,
                    'max_flow': max_flow,
                    'evacuation_time': phase_time,
                    'bottleneck_edge': bottleneck_edge,
                    'congested_edges': congested_edges[:20] if congested_edges else []
                })
                
                log_print(f"Phase {phase_num} evacuation time: {phase_time:.2f} hours ({phase_time*60:.1f} minutes)")
                cumulative_time += phase_time
                
                # Add to cumulative output
                cumulative_output.append(f"Phase {phase_num}:")
                cumulative_output.append(f"  Population: {phase_pop:,}")
                cumulative_output.append(f"  Evacuation time: {phase_time:.2f} hours ({phase_time*60:.1f} minutes)")
                cumulative_output.append(f"  Bottleneck flow: {max_flow:,} people/hour")
                if bottleneck_edge and F.has_edge(bottleneck_edge[0], bottleneck_edge[1]):
                    road_type = F[bottleneck_edge[0]][bottleneck_edge[1]].get('road_type', 'UNKNOWN')
                    cumulative_output.append(f"  Bottleneck road type: {road_type}")
                cumulative_output.append("")
                
            else:
                # If optimal solution not found, provide an estimate
                log_print(f"  Solver status: {status} - No optimal solution found.")
                
                # Estimate a reasonable evacuation time
                estimated_rate = 20000  # Conservative estimate
                estimated_time = phase_pop / estimated_rate
                
                cumulative_output.append(f"Phase {phase_num}: No feasible flow found")
                cumulative_output.append(f"  Estimated evacuation time: {estimated_time:.2f} hours (rough estimate)")
                cumulative_output.append("")
                
                cumulative_time += estimated_time
                
                # Add to phase results
                phase_results.append({
                    'phase': phase_num,
                    'population': phase_pop,
                    'max_flow': estimated_rate,
                    'evacuation_time': estimated_time,
                    'bottleneck_edge': None,
                    'notes': "Estimated time only - no feasible solution found"
                })
                
        except Exception as e:
            log_print(f"Error in Phase {phase_num}: {str(e)}")
            
            # Handle the error with estimated evacuation time
            estimated_rate = 20000  # Conservative estimate
            estimated_time = phase_pop / estimated_rate
            
            cumulative_output.append(f"Phase {phase_num}: Error - {str(e)}")
            cumulative_output.append(f"  Estimated evacuation time: {estimated_time:.2f} hours (estimate after error)")
            cumulative_output.append("")
            
            cumulative_time += estimated_time
            
            # Add estimate to phase results
            phase_results.append({
                'phase': phase_num,
                'population': phase_pop,
                'max_flow': estimated_rate,
                'evacuation_time': estimated_time,
                'bottleneck_edge': None,
                'notes': f"Error occurred: {str(e)}"
            })
        
        phase_time_taken = time.time() - phase_start_time
        log_print(f"Phase {phase_num} calculation took {phase_time_taken:.2f} seconds")
    
    # Write overall summary
    summary_path = os.path.join(output_dir, "phased_evacuation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Time-Phased Evacuation Analysis (OR-Tools)\n")
        f.write("======================================\n\n")
        f.write(f"Total Population: {total_population:,}\n")
        f.write(f"Evacuation Phases: {phases}\n")
        f.write(f"Total Evacuation Time: {cumulative_time:.2f} hours ({cumulative_time*60:.1f} minutes)\n\n")
        
        f.write("Phase-by-Phase Analysis:\n")
        f.write("=======================\n\n")
        
        for line in cumulative_output:
            f.write(line + "\n")
        
        # Add detailed population by phase
        f.write("\nDetailed Population by Phase:\n")
        for phase_idx, sources in enumerate(phase_sources):
            f.write(f"\nPhase {phase_idx+1} ZIP Codes:\n")
            for _, pop, zip_code in sources:
                f.write(f"  {zip_code}: {pop:,} people\n")
    
    log_print(f"\nTime-phased evacuation analysis completed in {time.time() - start_time:.2f} seconds")
    log_print(f"Total estimated evacuation time: {cumulative_time:.2f} hours ({cumulative_time*60:.1f} minutes)")
    log_print(f"Results saved to {summary_path}")
    
    return {
        'total_population': total_population,
        'total_evacuation_time': cumulative_time,
        'phase_results': phase_results
    }

def analyze_phased_evacuation_flow_only(G, output_dir, phases=12):
    """Analyze evacuation using a phased approach focused only on flow capacity"""
    log_print(f"Analyzing evacuation flow using {phases}-phase evacuation model with OR-Tools...")
    start_time = time.time()
    
    # Identify source (centroid) nodes with population data
    source_nodes = []
    sink_nodes = []
    
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "centroid" and data.get("demand", 0) < 0:
            population = -data.get("demand", 0)
            zip_code = data.get("zip_code", "unknown")
            source_nodes.append((node, population, zip_code))
        elif data.get("node_type") == "sink":  # Fixed syntax error with double parentheses
            sink_nodes.append(node)
    
    # Sort source nodes by population in descending order
    source_nodes.sort(key=lambda x: x[1], reverse=True)
    total_population = sum(pop for _, pop, _ in source_nodes)
    
    log_print(f"Planning evacuation for {total_population:,} people from {len(source_nodes)} zip codes")
    log_print(f"Dividing into {phases} evacuation waves")
    
    # Divide sources into phases
    sources_per_phase = len(source_nodes) // phases
    remainder = len(source_nodes) % phases
    
    phase_sources = []
    start_idx = 0
    
    for i in range(phases):
        extra = 1 if i < remainder else 0
        end_idx = start_idx + sources_per_phase + extra
        phase_sources.append(source_nodes[start_idx:end_idx])
        start_idx = end_idx
    
    # Process each phase
    phase_results = []
    cumulative_time = 0
    cumulative_output = []
    
    for phase_idx, sources in enumerate(phase_sources):
        phase_start_time = time.time()
        phase_num = phase_idx + 1
        log_print(f"\nProcessing Phase {phase_num} with {len(sources)} source nodes")
        
        # Create phase-specific graph copy
        F = G.copy()
        
        # Reset all demands
        for node in F.nodes():
            if 'demand' in F.nodes[node]:
                F.nodes[node]['demand'] = 0
        
        # Set demand only for current phase sources - REDUCE DEMAND BY 25%
        phase_pop = 0
        original_phase_pop = 0
        for node, pop, zip_code in sources:
            # Scale down the population to make the problem more feasible
            original_pop = pop
            original_phase_pop += original_pop
            scaled_pop = int(pop * 0.25)  # Try evacuating 25% at a time
            F.nodes[node]['demand'] = -scaled_pop
            phase_pop += scaled_pop
            log_print(f"  Phase {phase_num} source: {zip_code} with population {original_pop:,} (using {scaled_pop:,} for calculation)")
        
        # Create a super sink with adjusted capacity
        super_sink = f"super_sink_phase_{phase_num}"
        F.add_node(super_sink, demand=phase_pop)
        
        # Connect sinks to super sink with proper capacity
        for sink in sink_nodes:
            # Use 10x phase population for sink capacity to ensure it's not a bottleneck
            sink_capacity = phase_pop * 10
            F.add_edge(sink, super_sink, capacity=sink_capacity, weight=1)
            log_print(f"  Connecting sink {sink} to super sink with capacity {sink_capacity:,}")
        
        log_print(f"Phase {phase_num} original population: {original_phase_pop:,}")
        log_print(f"Phase {phase_num} scaled population for calculation: {phase_pop:,}")
        
        try:
            # Create the min cost flow solver
            mcf = min_cost_flow.SimpleMinCostFlow()
            
            # Create node maps for OR-Tools
            node_map = {}
            reverse_map = {}
            
            # Map nodes to integers for OR-Tools
            for i, node in enumerate(F.nodes()):
                node_map[node] = i
                reverse_map[i] = node
            
            # Add each node with demand
            for node, data in F.nodes(data=True):
                demand = data.get('demand', 0)
                if demand != 0:
                    mcf.set_node_supply(node_map[node], int(demand))
            
            # Add each edge as an arc - use existing capacities
            for u, v, data in F.edges(data=True):
                # Focus only on capacity - this is the key change
                capacity = int(data.get('capacity', 1000))
                
                # Increase centroid connections capacity to ensure feasibility
                if F.nodes[u].get('node_type') == 'centroid' or F.nodes[v].get('node_type') == 'centroid':
                    capacity = max(capacity, phase_pop * 2)
                
                # Use unit weights for simplicity
                weight = 1
                
                # Add the arc
                mcf.add_arc_with_capacity_and_unit_cost(
                    node_map[u], node_map[v], capacity, weight)
                
                # For undirected graphs, add reverse edge
                if not F.is_directed():
                    mcf.add_arc_with_capacity_and_unit_cost(
                        node_map[v], node_map[u], capacity, weight)
            
            # Find the min cost flow
            log_print("  Starting OR-Tools MinCostFlow computation...")
            computation_start = time.time()
            status = mcf.solve()
            computation_time = time.time() - computation_start
            log_print(f"  OR-Tools computation completed in {computation_time:.2f} seconds")
            
            if status == mcf.OPTIMAL:
                log_print("  Optimal solution found!")
                
                # Extract flow results
                max_flow = 0
                bottleneck_edge = None
                congested_edges = []
                used_flow_paths = 0
                
                for i in range(mcf.num_arcs()):
                    if mcf.flow(i) > 0:
                        # Get arc endpoints
                        tail = mcf.tail(i)
                        head = mcf.head(i)
                        flow = mcf.flow(i)
                        capacity = mcf.capacity(i)
                        
                        # Map back to original node IDs
                        orig_u = reverse_map[tail]
                        orig_v = reverse_map[head]
                        
                        # Skip super sink connections for bottleneck analysis
                        if orig_v != super_sink and orig_u != super_sink:
                            used_flow_paths += 1
                            
                            if flow > max_flow:
                                max_flow = flow
                                bottleneck_edge = (orig_u, orig_v)
                            
                            # Check for congestion
                            if capacity > 0 and flow / capacity > 0.8:
                                road_type = 'UNKNOWN'
                                if F.has_edge(orig_u, orig_v):
                                    road_type = F[orig_u][orig_v].get('road_type', 'UNKNOWN')
                                    
                                congested_edges.append((orig_u, orig_v, flow, capacity, flow/capacity, road_type))
                
                # Calculate evacuation time for this phase (using flow rate)
                # This assumes the bottleneck flow rate (people/hour) is the limiting factor
                scaled_phase_time = phase_pop / max(1, max_flow)
                
                # Adjust for the full population (multiply by 4 since we're doing 25%)
                phase_time = scaled_phase_time * 4
                
                log_print(f"  Used {used_flow_paths} flow paths")
                
                if bottleneck_edge:
                    u, v = bottleneck_edge
                    log_print(f"  Bottleneck: {u} → {v} with flow {max_flow:,} people/hour")
                    if F.has_edge(u, v):
                        road_type = F[u][v].get('road_type', 'UNKNOWN')
                        log_print(f"  Bottleneck road type: {road_type}")
                    
                    log_print(f"  Found {len(congested_edges)} congested road segments (>80% capacity)")
                    if congested_edges:
                        for i, (u, v, flow, capacity, ratio, road_type) in enumerate(
                                sorted(congested_edges, key=lambda x: x[4], reverse=True)[:5]):
                            log_print(f"    {i+1}. {u} → {v} ({road_type}): {flow:,}/{capacity:,} ({ratio*100:.1f}%)")
                
                # Store results
                phase_results.append({
                    'phase': phase_num,
                    'population': original_phase_pop,
                    'max_flow_rate': max_flow,
                    'evacuation_time': phase_time,
                    'bottleneck_edge': bottleneck_edge,
                    'congested_edges': congested_edges[:20] if congested_edges else []
                })
                
                log_print(f"Phase {phase_num} evacuation time: {phase_time:.2f} hours ({phase_time*60:.1f} minutes)")
                log_print(f"  Note: Time accounts for full population of {original_phase_pop:,} people")
                cumulative_time += phase_time
                
                # Add to cumulative output
                cumulative_output.append(f"Phase {phase_num}:")
                cumulative_output.append(f"  Population: {original_phase_pop:,}")
                cumulative_output.append(f"  Evacuation time: {phase_time:.2f} hours ({phase_time*60:.1f} minutes)")
                cumulative_output.append(f"  Bottleneck flow rate: {max_flow:,} people/hour")
                if bottleneck_edge and F.has_edge(bottleneck_edge[0], bottleneck_edge[1]):
                    road_type = F[bottleneck_edge[0]][bottleneck_edge[1]].get('road_type', 'UNKNOWN')
                    cumulative_output.append(f"  Bottleneck road type: {road_type}")
                cumulative_output.append("")
                
            else:
                # If optimal solution not found, provide an estimate
                log_print(f"  Solver status: {status} - No optimal solution found.")
                
                # Estimate a reasonable evacuation time based on a conservative flow rate
                estimated_rate = 20000  # Conservative estimate of people/hour
                estimated_time = original_phase_pop / estimated_rate
                
                cumulative_output.append(f"Phase {phase_num}: No feasible flow found")
                cumulative_output.append(f"  Estimated evacuation time: {estimated_time:.2f} hours (rough estimate)")
                cumulative_output.append("")
                
                cumulative_time += estimated_time
                
                # Add to phase results
                phase_results.append({
                    'phase': phase_num,
                    'population': original_phase_pop,
                    'estimated_flow_rate': estimated_rate,
                    'evacuation_time': estimated_time,
                    'bottleneck_edge': None,
                    'notes': "Estimated time only - no feasible solution found"
                })
                
        except Exception as e:
            log_print(f"Error in Phase {phase_num}: {str(e)}")
            
            # Handle the error with estimated evacuation time
            estimated_rate = 20000  # Conservative estimate of people/hour
            estimated_time = original_phase_pop / estimated_rate
            
            cumulative_output.append(f"Phase {phase_num}: Error - {str(e)}")
            cumulative_output.append(f"  Estimated evacuation time: {estimated_time:.2f} hours (estimate after error)")
            cumulative_output.append("")
            
            cumulative_time += estimated_time
            
            # Add estimate to phase results
            phase_results.append({
                'phase': phase_num,
                'population': original_phase_pop,
                'estimated_flow_rate': estimated_rate,
                'evacuation_time': estimated_time,
                'bottleneck_edge': None,
                'notes': f"Error occurred: {str(e)}"
            })
        
        phase_time_taken = time.time() - phase_start_time
        log_print(f"Phase {phase_num} calculation took {phase_time_taken:.2f} seconds")
    
    # Write overall summary
    summary_path = os.path.join(output_dir, "phased_evacuation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Time-Phased Evacuation Analysis (Flow Rate Only)\n")
        f.write("===========================================\n\n")
        f.write(f"Total Population: {total_population:,}\n")
        f.write(f"Evacuation Phases: {phases}\n")
        f.write(f"Total Evacuation Time: {cumulative_time:.2f} hours ({cumulative_time*60:.1f} minutes)\n\n")
        
        f.write("Phase-by-Phase Analysis:\n")
        f.write("=======================\n\n")
        
        for line in cumulative_output:
            f.write(line + "\n")
        
        # Add detailed population by phase
        f.write("\nDetailed Population by Phase:\n")
        for phase_idx, sources in enumerate(phase_sources):
            f.write(f"\nPhase {phase_idx+1} ZIP Codes:\n")
            for _, pop, zip_code in sources:
                f.write(f"  {zip_code}: {pop:,} people\n")
    
    log_print(f"\nTime-phased evacuation analysis completed in {time.time() - start_time:.2f} seconds")
    log_print(f"Total estimated evacuation time: {cumulative_time:.2f} hours ({cumulative_time*60:.1f} minutes)")
    log_print(f"Results saved to {summary_path}")
    
    return {
        'total_population': total_population,
        'total_evacuation_time': cumulative_time,
        'phase_results': phase_results
    }

def test_single_centroid_flow(G, output_dir, zip_code="33647"):
    """Test evacuation flow for a single centroid to verify the model works"""
    log_print(f"Testing evacuation flow for ZIP code {zip_code}")
    
    # Find the centroid node for this ZIP
    centroid_node = None
    population = 0
    
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "centroid" and data.get("zip_code") == zip_code:
            centroid_node = node
            population = -data.get("demand", 0)
            break
    
    if not centroid_node:
        log_print(f"Error: Could not find centroid for ZIP {zip_code}")
        return
    
    log_print(f"Found centroid node {centroid_node} with population {population:,}")
    
    # Get sink nodes
    sink_nodes = [node for node, data in G.nodes(data=True) 
                 if data.get("node_type") == "sink"]
    
    # Create a test graph copy
    F = G.copy()
    
    # Reset all demands
    for node in F.nodes():
        if 'demand' in F.nodes[node]:
            F.nodes[node]['demand'] = 0
    
    # Set demand for just this centroid - use 25% to ensure feasibility
    scaled_pop = int(population * 0.25)
    F.nodes[centroid_node]['demand'] = -scaled_pop
    
    # Create a super sink
    super_sink = "super_sink_test"
    F.add_node(super_sink, demand=scaled_pop)
    
    # Connect sinks to super sink
    for sink in sink_nodes:
        F.add_edge(sink, super_sink, capacity=scaled_pop*10, weight=1)
    
    # Create the min cost flow solver
    mcf = min_cost_flow.SimpleMinCostFlow()
    
    # Create node maps
    node_map = {}
    reverse_map = {}
    
    for i, node in enumerate(F.nodes()):
        node_map[node] = i
        reverse_map[i] = node
    
    # Add each node with demand
    for node, data in F.nodes(data=True):
        demand = data.get('demand', 0)
        if demand != 0:
            mcf.set_node_supply(node_map[node], int(demand))
    
    # Add each edge as an arc
    for u, v, data in F.edges(data=True):
        capacity = int(data.get('capacity', 1000))
        
        # Increase capacity for centroid connections
        if F.nodes[u].get('node_type') == 'centroid' or F.nodes[v].get('node_type') == 'centroid':
            capacity = max(capacity, scaled_pop * 2)
        
        weight = 1
        
        mcf.add_arc_with_capacity_and_unit_cost(
            node_map[u], node_map[v], capacity, weight)
        
        if not F.is_directed():
            mcf.add_arc_with_capacity_and_unit_cost(
                node_map[v], node_map[u], capacity, weight)
    
    # Solve
    log_print("Starting min-cost flow computation...")
    status = mcf.solve()
    
    if status == mcf.OPTIMAL:
        log_print("Optimal solution found!")
        
        # Extract results
        max_flow = 0
        bottleneck_edge = None
        
        for i in range(mcf.num_arcs()):
            if mcf.flow(i) > 0:
                tail = mcf.tail(i)
                head = mcf.head(i)
                flow = mcf.flow(i)
                
                orig_u = reverse_map[tail]
                orig_v = reverse_map[head]
                
                if orig_v != super_sink and orig_u != super_sink:
                    if flow > max_flow:
                        max_flow = flow
                        bottleneck_edge = (orig_u, orig_v)
        
        # Calculate evacuation time - adjust for full population
        scaled_evac_time = scaled_pop / max(1, max_flow)
        evac_time = scaled_evac_time * 4  # Full population
        
        log_print(f"Maximum flow: {max_flow:,} people/hour")
        log_print(f"Evacuation time: {evac_time:.2f} hours ({evac_time*60:.1f} minutes) for full population of {population:,}")
        
        if bottleneck_edge:
            u, v = bottleneck_edge
            road_type = "UNKNOWN"
            if F.has_edge(u, v):
                road_type = F[u][v].get('road_type', 'UNKNOWN')
            log_print(f"Bottleneck: {u} → {v} ({road_type})")
            
        return {
            'status': 'OPTIMAL',
            'population': population,
            'max_flow': max_flow,
            'evacuation_time': evac_time
        }
    else:
        log_print(f"Solver status: {status} - No optimal solution found")
        return {
            'status': f'NOT_OPTIMAL: {status}',
            'population': population
        }

def main():
    # Start timing for overall execution
    total_start_time = time.time()
    
    # Define file paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    routes_path = os.path.join(data_dir, "Evacuation_Routes_Tampa.geojson")
    centroids_path = os.path.join(data_dir, "zip_code_centroids.json")
    sink_nodes_path = os.path.join(data_dir, "sink_nodes.json")
    terminal_nodes_path = os.path.join(data_dir, "terminal_nodes.json")
    components_path = os.path.join(data_dir, "graph_components.png")
    results_dir = os.path.join(data_dir, "results")
    graph_pickle_path = os.path.join(data_dir, "evacuation_graph.pickle")
    population_data_path = os.path.join(data_dir, "Tampa-zip-codes-data.json")
    log_file_path = os.path.join(results_dir, "evacuation_log.txt")
    
    # Set up logging
    setup_logging(log_file_path)
    log_print(f"Starting evacuation analysis at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Track graph preparation time
    graph_prep_time = 0
    graph_prep_start = time.time()
    
    # Check if pickle exists and load it
    if os.path.exists(graph_pickle_path):
        # Load graph from pickle
        G = load_graph_from_pickle(graph_pickle_path)
        _, centroids_gdf, _ = load_data(routes_path, centroids_path, sink_nodes_path)
    else:
        # Load full data and create graph from scratch
        routes_gdf, centroids_gdf, sinks_gdf = load_data(routes_path, centroids_path, sink_nodes_path)
        
        # Create graph with improved connectivity
        G = create_connected_graph(routes_gdf, centroids_gdf, sinks_gdf, connection_threshold=150)
        
        # Connect sinks to terminals according to the terminal_nodes.json file
        G = connect_sinks_to_terminals(G, terminal_nodes_path, sink_nodes_path)
        
        # Save the newly created graph
        save_graph_to_pickle(G, graph_pickle_path)
    
    # Assign capacities and population data to the graph
    G = assign_road_capacities_and_population(G, population_data_path, sink_nodes_path)
    
    graph_prep_time = time.time() - graph_prep_start
    
    # Start timing pathfinding process
    pathfinding_start = time.time()
    
    # Process all centroids and find paths
    results = process_all_centroids(G, centroids_gdf, results_dir)
    
    # First, test with a single centroid
    log_print("\n=== TESTING SINGLE CENTROID FLOW ===\n")
    test_result = test_single_centroid_flow(G, results_dir, "33647")  # Test with largest population
    
    if test_result and test_result.get('status') == 'OPTIMAL':
        log_print("\n=== SINGLE CENTROID TEST SUCCESSFUL ===")
        log_print("Proceeding with full evacuation analysis...\n")
        
        # Run the full evacuation analysis
        flow_results = analyze_phased_evacuation_flow_only(G, results_dir, phases=12)
    else:
        log_print("\n=== SINGLE CENTROID TEST FAILED ===")
        log_print("Please check the graph structure and capacity settings.")
        
        # Try with an alternate single centroid
        log_print("\nTrying another ZIP code...")
        alt_test = test_single_centroid_flow(G, results_dir, "33511")  # Second largest population
        
        if alt_test and alt_test.get('status') == 'OPTIMAL':
            log_print("\n=== ALTERNATE CENTROID TEST SUCCESSFUL ===")
            log_print("Proceeding with full evacuation analysis...\n")
            flow_results = analyze_phased_evacuation_flow_only(G, results_dir, phases=12)
        else:
            log_print("\n=== MULTIPLE CENTROID TESTS FAILED ===")
            log_print("Fundamental issue with model structure detected.")
    
    pathfinding_time = time.time() - pathfinding_start
    total_time = time.time() - total_start_time
    
    # Add runtime information to the summary file
    summary_path = os.path.join(results_dir, "evacuation_summary.txt")
    
    with open(summary_path, 'a') as f:
        f.write("\n\nPerformance Summary\n")
        f.write("=================\n")
        f.write(f"Graph Preparation Time: {graph_prep_time:.2f} seconds ({graph_prep_time/60:.2f} minutes)\n")
        f.write(f"Pathfinding Time: {pathfinding_time:.2f} seconds ({pathfinding_time/60:.2f} minutes)\n")
        f.write(f"Total Run Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")
        f.write(f"Average Time Per Centroid: {pathfinding_time/len(centroids_gdf):.2f} seconds\n")

if __name__ == "__main__":
    main()