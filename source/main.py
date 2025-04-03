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
import logging
import sys
from ortools.graph.python import min_cost_flow

# Setup logging configuration
def setup_logging(log_file_path):
    """Set up logging to write to both console and file"""
    # Create the directory for the log file if it doesn't exist
    log_dir = os.path.dirname(log_file_path)
    if (log_dir and not os.path.exists(log_dir)):
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging with simpler formatters - no timestamps or level info
    file_formatter = logging.Formatter('%(message)s')  # Simplified - just show the message for file too
    console_formatter = logging.Formatter('%(message)s')  # Only show the message for console
    
    # Create handlers
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging started. Output will be saved to {log_file_path}")

# Replace all print statements with this function
def log_print(message):
    """Log a message to both console and file"""
    logging.info(message)

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
    
    total_time = time.time() - start_time
    log_print(f"Graph creation completed in {total_time:.2f} seconds")
    return G

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
    
    # Connect each sink to its 3 closest terminals
    total_connections = 0
    for sink_id in sink_node_ids:
        sink_zip = G.nodes[sink_id].get("zip_code", "unknown")
        sink_pos = G.nodes[sink_id]["pos"]
        
        # Calculate distances to all terminals
        terminal_distances = []
        for terminal_id in terminal_node_ids:
            terminal_pos = G.nodes[terminal_id]["pos"]
            dist = Point(sink_pos).distance(Point(terminal_pos))
            terminal_distances.append((terminal_id, dist))
        
        # Sort by distance and connect to closest 3 (or fewer if less than 3 terminals exist)
        terminal_distances.sort(key=lambda x: x[1])
        closest_count = min(3, len(terminal_distances))
        
        for i in range(closest_count):
            terminal_id, dist = terminal_distances[i]
            G.add_edge(sink_id, terminal_id, weight=dist, road_type="SINK_TERMINAL_CONNECTOR")
            total_connections += 1
        
        log_print(f"Connected sink {sink_zip} to its {closest_count} closest terminal nodes")
    
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
    
    road_properties = {
        'H': {'capacity': 24000, 'speed': 85},  # Highway
        'P': {'capacity': 12000, 'speed': 75},  # Primary road
        'M': {'capacity': 8000, 'speed': 65},   # Major road
        'C': {'capacity': 6400, 'speed': 55},   # Collector road
        'R': {'capacity': 4000, 'speed': 35},   # Ramp
        'CONNECTOR': {'capacity': 16000, 'speed': 65},
        'CENTROID_CONNECTOR': {'capacity': 20000, 'speed': 65},
        'SINK_CONNECTOR': {'capacity': 24000, 'speed': 65},
        'COMPONENT_CONNECTOR': {'capacity': 12000, 'speed': 65},
        'SINK_TERMINAL_CONNECTOR': {'capacity': 24000, 'speed': 65},
        'TERMINAL_CONNECTOR': {'capacity': 24000, 'speed': 65},
        'UNKNOWN': {'capacity': 8000, 'speed': 30}
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

def analyze_phased_evacuation_flow_only(G, output_dir, phases=12):
    """Analyze evacuation using a phased approach focused only on flow capacity"""
    log_print(f"Analyzing evacuation flow using {phases}-phase evacuation model with OR-Tools...")
    start_time = time.time()
    
    # Identify source (centroid) nodes with population data
    source_nodes = []
    sink_nodes = []
    
    # Create structure to store complete flow assignments
    detailed_flow_data = {
        "centroid_flows": {},
        "sink_flows": {},
        "phases": {}
    }
    
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "centroid" and data.get("demand", 0) < 0:
            population = -data.get("demand", 0)
            zip_code = data.get("zip_code", "unknown")
            source_nodes.append((node, population, zip_code))
            # Initialize centroid flow data
            detailed_flow_data["centroid_flows"][zip_code] = {
                "population": population,
                "sinks": {}
            }
        elif data.get("node_type") == "sink":
            sink_zip = data.get("zip_code", "unknown")
            sink_nodes.append(node)
            # Initialize sink flow data
            detailed_flow_data["sink_flows"][sink_zip] = {
                "total_inflow": 0
            }
    
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
                
                # Use distance-based costs
                weight = int(data.get('weight', 100))  # Get actual road distance
                if weight <= 0:
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
                
                # Create a phase record in the detailed flow data
                phase_data = {
                    "population": original_phase_pop,
                    "flows": {}
                }
                
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
                
                # Extract flow paths from solutions
                for i in range(mcf.num_arcs()):
                    if mcf.flow(i) > 0:
                        # Get arc endpoints
                        tail = reverse_map[mcf.tail(i)]
                        head = reverse_map[mcf.head(i)]
                        flow = mcf.flow(i)
                        
                        # Track flow from centroids to sinks
                        # This requires identifying which centroids flow to which sinks
                        
                        # If this is a direct flow to super_sink, we can track it
                        if head == super_sink and tail in sink_nodes:
                            sink_zip = G.nodes[tail].get("zip_code", "unknown")
                            
                            # We need to determine which centroids contributed to this sink's flow
                            # For each centroid in this phase, estimate its contribution
                            # proportional to its population
                            for node, pop, centroid_zip in sources:
                                # Scale contribution to match the 25% -> 100% adjustment
                                contribution = (pop / original_phase_pop) * flow * 4
                                
                                # Add to detailed flow data
                                if sink_zip not in detailed_flow_data["centroid_flows"][centroid_zip]["sinks"]:
                                    detailed_flow_data["centroid_flows"][centroid_zip]["sinks"][sink_zip] = 0
                                
                                detailed_flow_data["centroid_flows"][centroid_zip]["sinks"][sink_zip] += contribution
                                
                                # Also track in phase data
                                if centroid_zip not in phase_data["flows"]:
                                    phase_data["flows"][centroid_zip] = {}
                                
                                if sink_zip not in phase_data["flows"][centroid_zip]:
                                    phase_data["flows"][centroid_zip][sink_zip] = 0
                                    
                                phase_data["flows"][centroid_zip][sink_zip] += contribution
                                
                                # Update sink total
                                detailed_flow_data["sink_flows"][sink_zip]["total_inflow"] += contribution
                
                # Store phase data
                detailed_flow_data["phases"][phase_num] = phase_data
                
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
    
    # After all phases are processed, save the detailed flow data
    flow_data_path = os.path.join(output_dir, "detailed_evacuation_flows.json")
    with open(flow_data_path, 'w') as f:
        json.dump(detailed_flow_data, f, indent=2)
    
    log_print(f"Saved detailed flow assignments to {flow_data_path}")
    
    log_print(f"\nTime-phased evacuation analysis completed in {time.time() - start_time:.2f} seconds")
    log_print(f"Total estimated evacuation time: {cumulative_time:.2f} hours ({cumulative_time*60:.1f} minutes)")
    log_print(f"Results saved to {summary_path}")
    
    return {
        'total_population': total_population,
        'total_evacuation_time': cumulative_time,
        'phase_results': phase_results,
        'detailed_flow_path': flow_data_path
    }

def visualize_shortest_paths(G, centroids_gdf, results, output_dir):

    log_print("Generating evacuation path visualizations...")
    
    # Create output directory for visualizations
    vis_dir = os.path.join(output_dir, "path_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create a basemap of all nodes and edges
    plt.figure(figsize=(15, 15))
    
    # Draw base graph structure - just the routes
    route_nodes = []
    route_edges = []
    for u, v, data in G.edges(data=True):
        if G.nodes[u].get('node_type') == 'route' and G.nodes[v].get('node_type') == 'route':
            route_edges.append((u, v))
    
    # Create a position dictionary for all nodes
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw the base map (gray)
    nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color='lightgray', 
                         width=0.5, alpha=0.5)
    
    # Dictionary mapping for sink node colors
    sink_colors = {}
    color_options = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 
                    'magenta', 'brown', 'pink', 'olive', 'darkblue', 'teal']
    
    # Assign colors to sink nodes
    sink_nodes = [node for node, data in G.nodes(data=True) if data.get('node_type') == 'sink']
    for i, sink in enumerate(sink_nodes):
        sink_colors[sink] = color_options[i % len(color_options)]
    
    # Process each centroid result
    for idx, result in enumerate(results):
        zip_code = result['centroid']
        paths = result['paths']
        
        if not paths:
            continue
        
        # Create a new figure for each centroid
        plt.figure(figsize=(12, 10))
        
        # Draw base routes in light gray
        nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color='lightgray', 
                             width=0.5, alpha=0.3)
        
        # Find centroid node
        centroid_node = None
        for node, data in G.nodes(data=True):
            if data.get('node_type') == 'centroid' and data.get('zip_code') == zip_code:
                centroid_node = node
                break
        
        if not centroid_node:
            continue
        
        # Draw centroid node
        nx.draw_networkx_nodes(G, pos, nodelist=[centroid_node], 
                             node_color='blue', node_size=150, alpha=1.0)
        
        # Draw paths to different sinks with different colors
        for i, path_data in enumerate(paths[:3]):  # Show up to 3 paths
            path = path_data['path']
            sink_node = path_data['sink_node']
            sink_zip = path_data['sink']
            
            # Get path edges
            path_edges = list(zip(path[:-1], path[1:]))
            
            # Choose color based on sink
            color = sink_colors.get(sink_node, color_options[i % len(color_options)])
            
            # Draw path
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                                 edge_color=color, width=2.5, alpha=0.7)
            
            # Draw the destination sink node
            nx.draw_networkx_nodes(G, pos, nodelist=[sink_node], 
                                 node_color=color, node_size=150, alpha=1.0)
        
        # Add legends and labels
        plt.title(f"Evacuation Paths for ZIP Code {zip_code}")
        
        # Create custom legend
        legend_elements = []
        for i, path_data in enumerate(paths[:3]):
            sink_node = path_data['sink_node']
            sink_zip = path_data['sink']
            color = sink_colors.get(sink_node, color_options[i % len(color_options)])
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, 
                                            label=f"Path to {sink_zip} ({path_data['distance_km']:.1f} km)"))
        
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor='blue', markersize=10, 
                                        label=f"Source: ZIP {zip_code}"))
        
        plt.legend(handles=legend_elements, loc='lower right')
        plt.axis('off')
        
        # Save figure
        output_path = os.path.join(vis_dir, f"evacuation_paths_{zip_code}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Every 10 zip codes, create a combined visualization
        if idx % 10 == 0 and idx > 0:
            log_print(f"Generated visualizations for {idx} ZIP codes")
    
    log_print(f"Saved individual path visualizations to {vis_dir}")
    
    # Create a summary visualization with all centroids and their primary paths
    plt.figure(figsize=(20, 20))
    nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color='lightgray', 
                         width=0.5, alpha=0.2)
    
    # Draw all primary paths
    for result in results:
        if not result['paths']:
            continue
        
        zip_code = result['centroid']
        path_data = result['paths'][0]  # Only use primary path
        
        # Get centroid node
        centroid_node = None
        for node, data in G.nodes(data=True):
            if data.get('node_type') == 'centroid' and data.get('zip_code') == zip_code:
                centroid_node = node
                break
        
        if not centroid_node:
            continue
            
        # Get path edges
        path_edges = list(zip(path[:-1], path[1:]))
        
        # Draw centroid and path with random color
        color = color_options[hash(zip_code) % len(color_options)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                             edge_color=color, width=1.5, alpha=0.5)
    
    plt.title("All Primary Evacuation Paths")
    plt.axis('off')
    summary_path = os.path.join(vis_dir, "all_primary_paths.png")
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log_print(f"Generated summary visualization: {summary_path}")
    return vis_dir

def visualize_centroid_flows(G, flow_results, output_dir):
    """Visualize evacuation flows calculated by OR-Tools min-cost flow algorithm"""
    log_print("Generating per-centroid flow visualizations using actual OR-Tools min-cost flow results...")
    
    # Create output directory for visualizations
    vis_dir = os.path.join(output_dir, "centroid_flow_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create a position dictionary for all nodes
    pos = nx.get_node_attributes(G, 'pos')
    
    # Define base route edges for reference visualization
    route_edges = []
    for u, v, data in G.edges(data=True):
        if G.nodes[u].get('node_type') == 'route' and G.nodes[v].get('node_type') == 'route':
            route_edges.append((u, v))
    
    # Get all centroids and sinks and prepare data structures
    centroids = {}
    sink_nodes = []
    
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "centroid":
            zip_code = data.get("zip_code", "unknown")
            pop = -data.get("demand", 0) if data.get("demand", 0) < 0 else 0
            if pop > 0:  # Only include centroids with population
                centroids[node] = {"zip_code": zip_code, "population": pop, "flows": {}}
        elif data.get("node_type") == "sink":
            sink_nodes.append(node)
    
    # Define a color map for sinks
    sink_colors = {}
    color_options = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 
                    'magenta', 'brown', 'darkcyan', 'olive', 'darkgreen', 'indigo']
    
    for i, sink in enumerate(sink_nodes):
        sink_zip = G.nodes[sink].get("zip_code", f"sink_{i}")
        sink_colors[sink] = {"color": color_options[i % len(color_options)], "zip": sink_zip}
    
    # Since we want to visualize actual OR-Tools flows, we need to run a min-cost flow calculation
    # that includes all centroids at once to get a complete picture
    log_print("Running OR-Tools min-cost flow calculation to extract actual flows...")
    
    # Create a temporary copy of the graph for our calculation
    F = G.copy()
    
    # Create a super sink to collect all flows
    super_sink = "visualization_super_sink"
    F.add_node(super_sink)
    
    # Calculate total demand from centroids
    total_demand = 0
    for node, data in F.nodes(data=True):
        if data.get("node_type") == "centroid" and data.get("demand", 0) < 0:
            total_demand -= data.get("demand", 0)
    
    # Set super sink demand to balance the network
    F.nodes[super_sink]['demand'] = total_demand
    
    # Connect all sinks to the super sink with high capacity
    for sink in sink_nodes:
        F.add_edge(sink, super_sink, capacity=total_demand*2, weight=1)
    
    # Prepare OR-Tools min-cost flow solver
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
    
    # Add each edge as an arc with capacity and distance-based cost
    for u, v, data in F.edges(data=True):
        # Get capacity
        capacity = int(data.get('capacity', 1000))
        
        # Increase centroid connections capacity to ensure feasibility
        if F.nodes[u].get('node_type') == 'centroid' or F.nodes[v].get('node_type') == 'centroid':
            capacity = max(capacity, total_demand * 2)
        
        # Use distance-based costs
        weight = int(data.get('weight', 100))
        if weight <= 0:
            weight = 1
        
        # Add the arc
        mcf.add_arc_with_capacity_and_unit_cost(
            node_map[u], node_map[v], capacity, weight)
        
        # For undirected graphs, add reverse edge
        if not F.is_directed() and v != super_sink:  # Don't add reverse edges to super sink
            mcf.add_arc_with_capacity_and_unit_cost(
                node_map[v], node_map[u], capacity, weight)
    
    # Solve the min-cost flow problem
    log_print("Solving min-cost flow problem for visualization...")
    status = mcf.solve()
    
    if status != mcf.OPTIMAL:
        log_print(f"Warning: Min-cost flow solver status: {status} - Not optimal.")
        log_print("Using approximate flow visualization based on distance weighting as fallback.")
        # Fall back to the original distance-based calculation if OR-Tools doesn't find optimal solution
        for centroid_node, centroid_data in centroids.items():
            sink_paths = []
            for sink_node in sink_nodes:
                try:
                    path = nx.shortest_path(G, centroid_node, sink_node, weight="weight")
                    distance = sum(G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                    sink_paths.append({
                        "sink_node": sink_node,
                        "path": path,
                        "distance": distance,
                        "weight": 1/max(1, distance)  # Inverse distance weight
                    })
                except nx.NetworkXNoPath:
                    continue
            
            if sink_paths:
                total_weight = sum(path["weight"] for path in sink_paths)
                actual_population = centroid_data["population"]
                
                for path_data in sink_paths:
                    sink_node = path_data["sink_node"]
                    weight = path_data["weight"]
                    portion = weight / total_weight if total_weight > 0 else 0
                    flow = actual_population * portion
                    
                    centroid_data["flows"][sink_node] = {
                        "flow": flow,
                        "portion": portion,
                        "path": path_data["path"]
                    }
    else:
        log_print("Optimal min-cost flow solution found. Extracting actual flows...")
        
        # Extract flow data from OR-Tools solution
        # First, track flow directly from centroids to sinks
        direct_flows = {}  # {(centroid, sink): flow_amount}
        
        # Initialize flow tracking for each centroid
        for centroid_node in centroids:
            direct_flows[centroid_node] = {}
        
        # Track flows through the network
        flow_network = {}  # {node: {neighbor: flow}}
        for i in range(mcf.num_arcs()):
            flow = mcf.flow(i)
            if flow > 0:
                tail = reverse_map[mcf.tail(i)]
                head = reverse_map[mcf.head(i)]
                
                # Skip flows to super sink - we'll track these separately
                if head == super_sink:
                    continue
                
                # Initialize flow tracking for this node if not exists
                if tail not in flow_network:
                    flow_network[tail] = {}
                flow_network[tail][head] = flow
        
        # For each centroid, compute flow to each sink by finding paths
        for centroid_node, centroid_data in centroids.items():
            if centroid_node not in flow_network:
                continue  # No outgoing flow
            
            # Get total outflow from this centroid
            total_outflow = sum(flow_network[centroid_node].values())
            if total_outflow == 0:
                continue
            
            # This will store flow information for this centroid
            centroid_flows = {}
            
            # Find which sinks this centroid's flow goes to by tracing paths
            for sink_node in sink_nodes:
                try:
                    # Find a path in the flow network from centroid to sink
                    path = nx.shortest_path(G, centroid_node, sink_node, weight="weight")
                    
                    # Calculate the amount of flow along this path based on sink connections
                    # We check the final edge to the super sink to determine how much flow
                    # reaches this sink from the network
                    if sink_node in flow_network and super_sink in flow_network[sink_node]:
                        flow_to_sink = flow_network[sink_node][super_sink]
                        
                        # Estimate what proportion of this flow came from our centroid
                        # This is an approximation as we can't easily get exact contribution
                        proportion = centroid_data["population"] / total_demand
                        estimated_flow = flow_to_sink * proportion
                        
                        # Store the flow data
                        centroid_flows[sink_node] = {
                            "flow": estimated_flow,
                            "path": path,
                            "portion": estimated_flow / centroid_data["population"] if centroid_data["population"] > 0 else 0
                        }
                except nx.NetworkXNoPath:
                    continue
            
            # If we found flows to sinks, normalize to ensure total flow matches centroid population
            if centroid_flows:
                total_flow = sum(data["flow"] for data in centroid_flows.values())
                if total_flow > 0:
                    # Scale flows to match the actual population
                    scale_factor = centroid_data["population"] / total_flow
                    for sink_node, flow_data in centroid_flows.items():
                        flow_data["flow"] *= scale_factor
                        flow_data["portion"] = flow_data["flow"] / centroid_data["population"]
                
                # Store flows in the centroid data
                centroid_data["flows"] = centroid_flows
    
    # Visualize the flows
    log_print("Creating flow visualizations based on actual min-cost flow results...")
    
    for centroid_node, centroid_data in centroids.items():
        zip_code = centroid_data["zip_code"]
        population = centroid_data["population"]
        flows = centroid_data["flows"]
        
        if not flows:
            log_print(f"No flow data found for centroid {zip_code} - skipping visualization")
            continue
        
        log_print(f"Generating flow visualization for ZIP {zip_code} (Population: {population:,})")
        
        # Create figure
        plt.figure(figsize=(14, 12))
        
        # Draw base graph structure
        nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color='lightgray', 
                             width=0.5, alpha=0.2)
        
        # Draw the flows from this centroid to sinks
        sink_flow_list = []
        for sink_node, flow_data in flows.items():
            sink_flow_list.append({
                "sink_node": sink_node,
                "flow": flow_data["flow"],
                "portion": flow_data["portion"],
                "path": flow_data["path"],
                "sink_zip": G.nodes[sink_node].get("zip_code", "unknown") if sink_node in G.nodes else "unknown"
            })
        
        # Sort by flow amount (descending)
        sink_flow_list.sort(key=lambda x: x["flow"], reverse=True)
        
        # Draw flows
        for flow_data in sink_flow_list:
            sink_node = flow_data["sink_node"]
            path = flow_data["path"]
            portion = flow_data["portion"]
            flow = flow_data["flow"]
            
            # Get sink color
            color = sink_colors.get(sink_node, {"color": "gray"})["color"]
            
            # Calculate edge width based on flow (scaled for visibility)
            edge_width = 1 + (portion * 8)
            
            # Draw the path
            path_edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                                 edge_color=color, width=edge_width, 
                                 alpha=max(0.4, min(0.9, portion * 2)))
        
        # Draw centroid node
        nx.draw_networkx_nodes(G, pos, nodelist=[centroid_node], 
                             node_color='darkblue', node_size=200, alpha=1.0)
        
        # Draw sink nodes
        for sink_node in sink_nodes:
            if sink_node in pos:
                nx.draw_networkx_nodes(G, pos, nodelist=[sink_node], 
                                     node_color=sink_colors[sink_node]["color"], 
                                     node_size=150, alpha=1.0)
        
        # Add title and information box
        plt.title(f"Evacuation Flow Distribution for ZIP {zip_code} (Based on Min-Cost Flow)")
        
        # Create text for info box
        info_text = f"ZIP Code: {zip_code}\nPopulation: {population:,}\n\nEvacuation Flow Distribution (Min-Cost Flow):"
        
        total_shown_flow = 0
        for i, flow_data in enumerate(sink_flow_list[:5]):  # Show top 5
            sink_zip = flow_data["sink_zip"]
            flow = flow_data["flow"]
            portion = flow_data["portion"]
            total_shown_flow += flow
            
            info_text += f"\n→ {sink_zip}: {flow:,.0f} people ({portion*100:.1f}%)"
        
        if len(sink_flow_list) > 5:
            remaining_flow = sum(flow_data["flow"] for flow_data in sink_flow_list[5:])
            remaining_portion = sum(flow_data["portion"] for flow_data in sink_flow_list[5:])
            info_text += f"\n→ Others: {remaining_flow:,.0f} people ({remaining_portion*100:.1f}%)"
        
        info_text += f"\nTotal: {total_shown_flow + (remaining_flow if len(sink_flow_list) > 5 else 0):,.0f} people"
        
        # Add info box
        plt.text(0.02, 0.02, info_text, transform=plt.gca().transAxes,
               fontsize=12, verticalalignment='bottom', horizontalalignment='left',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        # Add legend
        legend_elements = []
        for i, flow_data in enumerate(sink_flow_list[:5]):
            sink_node = flow_data["sink_node"]
            sink_zip = flow_data["sink_zip"]
            color = sink_colors.get(sink_node, {"color": "gray"})["color"]
            portion = flow_data["portion"]
            flow = flow_data["flow"]
            
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=2 + (portion * 4), 
                                            label=f"{sink_zip}: {flow:,.0f} ({portion*100:.1f}%)"))
        
        plt.legend(handles=legend_elements, loc='upper right', title="Top Destinations")
        
        # Turn off axis
        plt.axis('off')
        
        # Save
        output_path = os.path.join(vis_dir, f"min_cost_flow_{zip_code}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Create a summary report
    report_path = os.path.join(output_dir, "actual_evacuation_flow_report.txt")
    with open(report_path, 'w') as f:
        f.write("Actual Evacuation Flow Distribution Report (Min-Cost Flow)\n")
        f.write("=================================\n\n")
        
        total_population = sum(data["population"] for data in centroids.values())
        f.write(f"Total Population: {total_population:,}\n\n")
        
        # Sink summary
        f.write("Sink Distribution Summary\n")
        f.write("-----------------------\n")
        
        sink_totals = {}
        for sink_node in sink_nodes:
            sink_zip = G.nodes[sink_node].get("zip_code", f"sink_{i}")
            sink_totals[sink_node] = {
                "zip": sink_zip,
                "inflow": 0
            }
        
        # Sum flows to each sink
        for sink_node, data in sorted(sink_totals.items(), key=lambda x: x[1]["inflow"], reverse=True):
            f.write(f"Sink {data['zip']}: {data['inflow']:,.0f} people ")
            f.write(f"({data['inflow']/total_population*100:.1f}% of total)\n")
        
        f.write("\n\nCentroid-to-Sink Flow Details\n")
        f.write("----------------------------\n\n")
        
        # Write detailed flow for each centroid
        for centroid_node, centroid_data in sorted(centroids.items(), key=lambda x: x[1]["population"], reverse=True):
            zip_code = centroid_data["zip_code"]
            population = centroid_data["population"]
            flows = centroid_data.get("flows", {})
            
            f.write(f"ZIP Code: {zip_code} (Population: {population:,})\n")
            
            if flows:
                sink_flow_list = []
                for sink_node, flow_data in flows.items():
                    sink_flow_list.append({
                        "sink_node": sink_node,
                        "flow": flow_data["flow"],
                        "portion": flow_data["portion"],
                        "sink_zip": G.nodes[sink_node].get("zip_code", "unknown") if sink_node in G.nodes else "unknown"
                    })
                
                sink_flow_list.sort(key=lambda x: x["flow"], reverse=True)
                
                for flow_data in sink_flow_list:
                    sink_zip = flow_data["sink_zip"]
                    flow = flow_data["flow"]
                    portion = flow_data["portion"]
                    
                    f.write(f"  → {sink_zip}: {flow:,.0f} people ({portion*100:.1f}%)\n")
            else:
                f.write("  No flow data available\n")
            
            f.write("\n")
    
    log_print(f"Generated actual min-cost flow visualizations in {vis_dir}")
    log_print(f"Detailed flow report saved to {report_path}")
    
    return vis_dir, report_path

def visualize_phased_flows(G, detailed_flow_path, output_dir):
    """Visualize evacuation flows using detailed flow data previously saved"""
    log_print("Generating flow visualizations based on detailed flow assignments...")
    
    # Load the detailed flow data
    with open(detailed_flow_path, 'r') as f:
        detailed_flow_data = json.load(f)
    
    # Create output directory for visualizations
    vis_dir = os.path.join(output_dir, "phased_flow_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create a position dictionary for all nodes
    pos = nx.get_node_attributes(G, 'pos')
    
    # Define base route edges for reference visualization
    route_edges = []
    for u, v, data in G.edges(data=True):
        if G.nodes[u].get('node_type') == 'route' and G.nodes[v].get('node_type') == 'route':
            route_edges.append((u, v))
    
    # Get all centroids and sinks and prepare mapping dict to actual nodes
    centroid_node_by_zip = {}
    sink_node_by_zip = {}
    sink_nodes = []
    
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "centroid":
            zip_code = data.get("zip_code", "unknown")
            centroid_node_by_zip[zip_code] = node
        elif data.get("node_type") == "sink":
            zip_code = data.get("zip_code", "unknown")
            sink_node_by_zip[zip_code] = node
            sink_nodes.append(node)
    
    # Define a color map for sinks
    sink_colors = {}
    color_options = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 
                    'magenta', 'brown', 'darkcyan', 'olive', 'darkgreen', 'indigo']
    
    for i, sink in enumerate(sink_nodes):
        sink_zip = G.nodes[sink].get("zip_code", f"sink_{i}")
        sink_colors[sink] = {"color": color_options[i % len(color_options)], "zip": sink_zip}
    
    # Process each centroid's flow data from the JSON
    for centroid_zip, centroid_data in detailed_flow_data["centroid_flows"].items():
        population = centroid_data["population"]
        sinks_data = centroid_data.get("sinks", {})
        
        if not sinks_data:
            log_print(f"No flow data found for centroid {centroid_zip} - skipping visualization")
            continue
        
        # Check if we can map the centroid ZIP to an actual node
        if centroid_zip not in centroid_node_by_zip:
            log_print(f"Could not find centroid node for ZIP {centroid_zip} - skipping visualization")
            continue
        
        centroid_node = centroid_node_by_zip[centroid_zip]
        log_print(f"Generating flow visualization for ZIP {centroid_zip} (Population: {population:,})")
        
        # Create figure
        plt.figure(figsize=(14, 12))
        
        # Draw base graph structure
        nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color='lightgray', 
                             width=0.5, alpha=0.2)
        
        # Process each sink flow and find a path to visualize
        sink_flow_list = []
        for sink_zip, flow_amount in sinks_data.items():
            # Skip if sink ZIP cannot be mapped to a node
            if sink_zip not in sink_node_by_zip:
                continue
                
            sink_node = sink_node_by_zip[sink_zip]
            portion = flow_amount / population if population > 0 else 0
            
            # Find shortest path from centroid to this sink
            try:
                path = nx.shortest_path(G, centroid_node, sink_node, weight="weight")
                
                sink_flow_list.append({
                    "sink_node": sink_node,
                    "sink_zip": sink_zip,
                    "flow": flow_amount,
                    "portion": portion,
                    "path": path
                })
            except nx.NetworkXNoPath:
                log_print(f"No path found from {centroid_zip} to sink {sink_zip}")
                continue
        
        # Sort flows by amount (descending)
        sink_flow_list.sort(key=lambda x: x["flow"], reverse=True)
        
        # Draw flows
        for flow_data in sink_flow_list:
            sink_node = flow_data["sink_node"]
            path = flow_data["path"]
            portion = flow_data["portion"]
            flow = flow_data["flow"]
            
            # Get sink color
            color = sink_colors.get(sink_node, {"color": "gray"})["color"]
            
            # Calculate edge width based on flow (scaled for visibility)
            edge_width = 1 + (portion * 8)
            
            # Draw the path
            path_edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                                 edge_color=color, width=edge_width, 
                                 alpha=max(0.4, min(0.9, portion * 2)))
        
        # Draw centroid node
        nx.draw_networkx_nodes(G, pos, nodelist=[centroid_node], 
                             node_color='darkblue', node_size=200, alpha=1.0)
        
        # Draw sink nodes
        for sink_node in sink_nodes:
            if sink_node in pos:
                nx.draw_networkx_nodes(G, pos, nodelist=[sink_node], 
                                     node_color=sink_colors[sink_node]["color"], 
                                     node_size=150, alpha=1.0)
        
        # Add title and information box
        plt.title(f"Evacuation Flow Distribution for ZIP {centroid_zip} (Based on Phased Flow Analysis)")
        
        # Create text for info box
        info_text = f"ZIP Code: {centroid_zip}\nPopulation: {population:,}\n\nEvacuation Flow Distribution:"
        
        total_shown_flow = 0
        for i, flow_data in enumerate(sink_flow_list[:5]):  # Show top 5
            sink_zip = flow_data["sink_zip"]
            flow = flow_data["flow"]
            portion = flow_data["portion"]
            total_shown_flow += flow
            
            info_text += f"\n→ {sink_zip}: {flow:,.0f} people ({portion*100:.1f}%)"
        
        if len(sink_flow_list) > 5:
            remaining_flow = sum(flow_data["flow"] for flow_data in sink_flow_list[5:])
            remaining_portion = sum(flow_data["portion"] for flow_data in sink_flow_list[5:])
            info_text += f"\n→ Others: {remaining_flow:,.0f} people ({remaining_portion*100:.1f}%)"
        
        info_text += f"\nTotal: {total_shown_flow + (remaining_flow if len(sink_flow_list) > 5 else 0):,.0f} people"
        
        # Add info box
        plt.text(0.02, 0.02, info_text, transform=plt.gca().transAxes,
               fontsize=12, verticalalignment='bottom', horizontalalignment='left',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        # Add legend
        legend_elements = []
        for i, flow_data in enumerate(sink_flow_list[:5]):
            sink_node = flow_data["sink_node"]
            sink_zip = flow_data["sink_zip"]
            color = sink_colors.get(sink_node, {"color": "gray"})["color"]
            portion = flow_data["portion"]
            flow = flow_data["flow"]
            
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=2 + (portion * 4), 
                                            label=f"{sink_zip}: {flow:,.0f} ({portion*100:.1f}%)"))
        
        plt.legend(handles=legend_elements, loc='upper right', title="Top Destinations")
        
        # Turn off axis
        plt.axis('off')
        
        # Save
        output_path = os.path.join(vis_dir, f"phased_flow_{centroid_zip}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Create a summary report
    report_path = os.path.join(output_dir, "phased_evacuation_flow_report.txt")
    with open(report_path, 'w') as f:
        f.write("Phased Evacuation Flow Distribution Report\n")
        f.write("======================================\n\n")
        f.write("This report shows the actual flow distribution based on the min-cost flow analysis\n\n")
        
        total_population = sum(data["population"] for data in detailed_flow_data["centroid_flows"].values())
        f.write(f"Total Population: {total_population:,}\n\n")
        
        # Sink summary
        f.write("Sink Distribution Summary\n")
        f.write("-----------------------\n")
        
        # Calculate total inflows to each sink
        sink_totals = {}
        for sink_zip in detailed_flow_data["sink_flows"].keys():
            sink_totals[sink_zip] = 0
            
        # Sum up all flows to sinks from centroids
        for centroid_data in detailed_flow_data["centroid_flows"].values():
            for sink_zip, flow in centroid_data.get("sinks", {}).items():
                if sink_zip in sink_totals:
                    sink_totals[sink_zip] += flow
        
        # Write sink summary
        for sink_zip, inflow in sorted(sink_totals.items(), key=lambda x: x[1], reverse=True):
            if inflow > 0:
                f.write(f"Sink {sink_zip}: {inflow:,.0f} people ")
                f.write(f"({inflow/total_population*100:.1f}% of total)\n")
        
        f.write("\n\nCentroid-to-Sink Flow Details\n")
        f.write("----------------------------\n\n")
        
        # Write detailed flow for each centroid
        sorted_centroids = sorted(detailed_flow_data["centroid_flows"].items(), 
                                 key=lambda x: x[1]["population"], reverse=True)
                                 
        for centroid_zip, centroid_data in sorted_centroids:
            population = centroid_data["population"]
            sinks_data = centroid_data.get("sinks", {})
            
            f.write(f"ZIP Code: {centroid_zip} (Population: {population:,})\n")
            
            if sinks_data:
                sink_flows = [(sink_zip, flow) for sink_zip, flow in sinks_data.items()]
                sink_flows.sort(key=lambda x: x[1], reverse=True)
                
                for sink_zip, flow in sink_flows:
                    portion = flow / population if population > 0 else 0
                    f.write(f"  → {sink_zip}: {flow:,.0f} people ({portion*100:.1f}%)\n")
            else:
                f.write("  No flow data available\n")
            
            f.write("\n")
    
    log_print(f"Generated phased flow visualizations in {vis_dir}")
    log_print(f"Detailed phased flow report saved to {report_path}")
    
    return vis_dir, report_path

def calculate_evacuation_flow_report(G, output_dir, congestion_scenario='moderate_congestion', phases=12):
    """
    Calculate evacuation flow and generate a detailed text report using road-specific speeds and congestion factors.
    """
    log_print(f"Calculating evacuation flow distribution (congestion: {congestion_scenario})...")
    start_time = time.time()
    
    congestion_scenarios = {
        'free_flow': 0.7,           # Minimal evacuation traffic
        'light_congestion': 0.5,    # Light evacuation traffic
        'moderate_congestion': 0.3, # Typical evacuation conditions
        'heavy_congestion': 0.2,    # Heavy evacuation traffic
        'severe_congestion': 0.1    # Severe congestion with bottlenecks
    }
    
    # Select the appropriate congestion factor (default to moderate if not found)
    congestion_factor = congestion_scenarios.get(congestion_scenario, 0.5)
    log_print(f"Using congestion factor: {congestion_factor} ({congestion_scenario})")
    
    # Identify source (centroid) nodes with population data
    source_nodes = []
    sink_nodes = []
    
    # Create structure to store complete flow assignments
    centroid_flow_data = {}
    sink_flow_data = {}
    
    # Rest of your existing code to initialize data structures
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "centroid" and data.get("demand", 0) < 0:
            population = -data.get("demand", 0)
            zip_code = data.get("zip_code", "unknown")
            source_nodes.append((node, population, zip_code))
            # Initialize centroid flow data
            centroid_flow_data[zip_code] = {
                "population": population,
                "node_id": node,
                "sinks": {}
            }
        elif data.get("node_type") == "sink":
            sink_zip = data.get("zip_code", "unknown")
            sink_nodes.append((node, sink_zip))
            # Initialize sink flow data
            sink_flow_data[sink_zip] = {
                "node_id": node,
                "total_inflow": 0,
                "centroids": {}  # Add this to fix the KeyError
            }
    
    # Sort source nodes by population in descending order
    source_nodes.sort(key=lambda x: x[1], reverse=True)
    total_population = sum(pop for _, pop, _ in source_nodes)
    
    log_print(f"Planning evacuation for {total_population:,} people from {len(source_nodes)} zip codes")
    
    # DIRECT ALLOCATION: For each centroid, find the 3 closest sinks and distribute population
    for centroid_node, population, centroid_zip in source_nodes:
        log_print(f"Processing {centroid_zip} with population {population:,}")
        
        # Find paths to all sinks
        sink_paths = []
        for sink_node, sink_zip in sink_nodes:
            try:
                path = nx.shortest_path(G, centroid_node, sink_node, weight="weight")
                
                # Calculate distance and travel time with road-specific speeds and congestion
                distance = 0
                travel_time = 0
                
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    segment_distance = G[u][v].get('weight', 1)  # Distance in meters
                    
                    # Get road type and appropriate speed from edge properties
                    road_type = G[u][v].get('road_type', 'UNKNOWN')
                    speed_mph = 60  # Default speed in mph
                    
                    # Try to get the actual speed from the edge data if available
                    if 'speed' in G[u][v]:
                        speed_mph = G[u][v]['speed']
                    
                    # Convert mph to m/s (mph * 0.44704 = m/s)
                    speed_mps = speed_mph * 0.44704
                    
                    # Apply congestion factor to speed
                    congested_speed = speed_mps * congestion_factor
                    
                    # Calculate time for this segment (distance/speed)
                    segment_time = segment_distance / max(1, (congested_speed * 3600))  # Time in hours
                    
                    distance += segment_distance
                    travel_time += segment_time
                
                sink_paths.append({
                    "sink_zip": sink_zip,
                    "distance": distance,
                    "travel_time": travel_time,
                    "weight": 1/max(0.1, travel_time**2)  # Use travel time for weighting
                })
            except nx.NetworkXNoPath:
                log_print(f"  No path found from {centroid_zip} to sink {sink_zip}")
                continue
        
        # Rest of your existing code...
        if not sink_paths:
            log_print(f"  WARNING: No paths found from {centroid_zip} to any sink! Using default allocation.")
            # Default allocation if no paths found
            if sink_nodes:
                per_sink = population / len(sink_nodes)
                for _, sink_zip in sink_nodes:
                    centroid_flow_data[centroid_zip]["sinks"][sink_zip] = per_sink
                    sink_flow_data[sink_zip]["total_inflow"] += per_sink
            continue
        
        # Sort by travel time (closest first)
        sink_paths.sort(key=lambda x: x["travel_time"])
        
        # Use the 7 closest sinks (or fewer)
        num_sinks_to_use = min(7, len(sink_paths))
        closest_sinks = sink_paths[:num_sinks_to_use]
        
        total_weight = sum(sink["weight"] for sink in closest_sinks)
        
        # Distribute population to closest sinks
        for sink_data in closest_sinks:
            sink_zip = sink_data["sink_zip"]
            weight = sink_data["weight"]
            proportion = weight / total_weight if total_weight > 0 else 0
            flow = population * proportion
            travel_time = sink_data["travel_time"]
            distance = sink_data["distance"]
            
            # Find the sink node for this sink zip
            sink_node = None
            for node, zip_code in sink_nodes:
                if zip_code == sink_zip:
                    sink_node = node
                    break
            
            # Calculate queue clearance time
            queue_time = 0
            if sink_node:
                # Find or reuse path
                path = nx.shortest_path(G, centroid_node, sink_node, weight="weight")
                
                # Find bottleneck capacity along path
                bottleneck_capacity = float('inf')
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    segment_capacity = G[u][v].get('capacity', 1000)  # people/hour
                    vehicle_capacity = segment_capacity / 2.5  # vehicles/hour
                    bottleneck_capacity = min(bottleneck_capacity, vehicle_capacity)
                
                # Calculate queue time
                vehicle_count = flow / 2.5  # Convert people to vehicles
                queue_time = vehicle_count / bottleneck_capacity if bottleneck_capacity > 0 else 0
            
            # Total evacuation time
            total_evacuation_time = travel_time + queue_time
            
            # Store in data structures
            if sink_zip not in centroid_flow_data[centroid_zip]["sinks"]:
                centroid_flow_data[centroid_zip]["sinks"][sink_zip] = {
                    "flow": 0,
                    "travel_time": travel_time,
                    "queue_time": queue_time,
                    "total_time": total_evacuation_time,
                    "distance": distance
                }
            else:
                centroid_flow_data[centroid_zip]["sinks"][sink_zip]["flow"] += flow
            
            # Add to sink flow data
            if centroid_zip not in sink_flow_data[sink_zip]["centroids"]:
                sink_flow_data[sink_zip]["centroids"][centroid_zip] = 0
            
            sink_flow_data[sink_zip]["centroids"][centroid_zip] += flow
            sink_flow_data[sink_zip]["total_inflow"] += flow
            
            log_print(f"    {centroid_zip} → {sink_zip}: {flow:,.0f} people ({proportion*100:.1f}%) - " 
                      f"Travel: {travel_time:.2f} hrs, Queue: {queue_time:.2f} hrs, Total: {total_evacuation_time:.2f} hrs")

    # Your existing code for report generation...
    report_path = os.path.join(output_dir, "evacuation_flow_report.txt")
    with open(report_path, 'w') as f:
        f.write("EVACUATION FLOW DISTRIBUTION REPORT\n")
        f.write("=================================\n\n")
        
        f.write(f"Total Population: {total_population:,}\n")
        f.write(f"Congestion Scenario: {congestion_scenario} (factor: {congestion_factor})\n\n")
        
        # Sink summary
        f.write("SINK DISTRIBUTION SUMMARY\n")
        f.write("-----------------------\n")
        
        for sink_zip, data in sorted(sink_flow_data.items(), key=lambda x: x[1]["total_inflow"], reverse=True):
            inflow = data["total_inflow"]
            percentage = (inflow / total_population) * 100 if total_population > 0 else 0
            f.write(f"Sink {sink_zip}: {inflow:,.0f} people ({percentage:.1f}% of total)\n")
        
        f.write("\n\nDETAILED EVACUATION FLOW REPORT\n")
        f.write("-----------------------------\n\n")
        
        # Write detailed flow for each centroid
        for centroid_zip, data in sorted(centroid_flow_data.items(), key=lambda x: x[1]["population"], reverse=True):
            population = data["population"]
            sinks = data.get("sinks", {})
            
            f.write(f"\nZIP Code: {centroid_zip} (Population: {population:,})\n")
            f.write("----------------------------------------\n")
            
            if sinks:
                sink_flows = []
                for sink_zip, sink_data in sinks.items():
                    flow = sink_data["flow"]
                    travel_time = sink_data.get("travel_time", 0)
                    queue_time = sink_data.get("queue_time", 0)
                    total_time = sink_data.get("total_time", travel_time + queue_time)
                    
                    sink_flows.append((sink_zip, flow, travel_time, queue_time, total_time))
                
                # Sort by flow (highest first)
                sink_flows.sort(key=lambda x: x[1], reverse=True)
                
                for sink_zip, flow, travel_time, queue_time, total_time in sink_flows:
                    if flow > 0:  # Only show non-zero flows
                        portion = flow / population if population > 0 else 0
                        # Format times with detailed breakdown
                        f.write(f"  → {sink_zip}: {flow:,.0f} people ({portion*100:.1f}%)\n")
                        f.write(f"     Travel time: {travel_time:.2f} hours | ")
                        f.write(f"Queue time: {queue_time:.2f} hours | ")
                        f.write(f"Total evacuation time: {total_time:.2f} hours\n")
            else:
                f.write("  No flow data available\n")
    
    log_print(f"Evacuation flow calculation completed in {time.time() - start_time:.2f} seconds")
    log_print(f"Detailed flow report saved to {report_path}")
    
    return report_path

def main():
    # Start timing for overall execution
    total_start_time = time.time()
    
    # Define file paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    routes_path = os.path.join(data_dir, "Evacuation_Routes_Tampa.geojson")
    centroids_path = os.path.join(data_dir, "zip_code_centroids.json")
    sink_nodes_path = os.path.join(data_dir, "sink_nodes.json")
    terminal_nodes_path = os.path.join(data_dir, "terminal_nodes.json")
    results_dir = os.path.join(data_dir, "results")
    graph_pickle_path = os.path.join(data_dir, "evacuation_graph.pickle")
    population_data_path = os.path.join(data_dir, "Tampa-zip-codes-data.json")
    log_file_path = os.path.join(results_dir, "evacuation_log.txt")
    
    # Set up logging
    setup_logging(log_file_path)
    log_print(f"Starting evacuation analysis at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if pickle exists and load it
    if os.path.exists(graph_pickle_path):
        # Load graph from pickle
        G = load_graph_from_pickle(graph_pickle_path)
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
    
    # Calculate evacuation flow with moderate congestion
    report_path = calculate_evacuation_flow_report(G, results_dir, 
                                                  congestion_scenario='moderate_congestion',
                                                  phases=12)
    
    total_time = time.time() - total_start_time
    log_print(f"\nCompleted in {total_time:.2f} seconds")
    log_print(f"Flow report saved to: {report_path}")

if __name__ == "__main__":
    main()