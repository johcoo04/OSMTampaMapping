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
    file_formatter = logging.Formatter('%(message)s')
    console_formatter = logging.Formatter('%(message)s')
    
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

def calculate_min_cost_flow(G, output_dir):
    """Calculate optimal evacuation flows using a simplified bi-level approach with optimized path finding."""
    log_print("Calculating optimal evacuation paths using bi-level min-cost flow approach...")
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Identify source/sink nodes and prepare data structures
    source_nodes = []
    sink_nodes = []
    
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "centroid" and data.get("demand", 0) < 0:
            population = -data.get("demand", 0)
            zip_code = data.get("zip_code", "unknown")
            source_nodes.append((node, population, zip_code))
        elif data.get("node_type") == "sink":
            sink_zip = data.get("zip_code", "unknown")
            capacity = data.get("capacity", 2000000)
            sink_nodes.append((node, sink_zip, capacity))
    
    total_population = sum(pop for _, pop, _ in source_nodes)
    total_sink_capacity = sum(cap for _, _, cap in sink_nodes)
    
    log_print(f"Planning optimal evacuation for {total_population:,} people from {len(source_nodes)} zip codes")
    log_print(f"Total sink capacity: {total_sink_capacity:,} (should be >= population)")
    
    # LEVEL 1: Create a simplified bipartite graph directly connecting sources to sinks
    log_print("Creating simplified evacuation network...")
    F = nx.DiGraph()
    
    # Add source nodes with demands (negative = supply)
    for node_id, pop, zip_code in source_nodes:
        F.add_node(node_id, demand=-pop, population=pop, zip_code=zip_code, node_type="centroid")
    
    # Add sink nodes
    for node_id, sink_zip, capacity in sink_nodes:
        F.add_node(node_id, capacity=capacity, zip_code=sink_zip, node_type="sink")
    
    # Add a super sink
    super_sink = "evacuation_super_sink"
    F.add_node(super_sink, demand=total_population, node_type="super_sink")
    
    # Connect each sink to super sink
    for sink_node, sink_zip, capacity in sink_nodes:
        F.add_edge(sink_node, super_sink, capacity=capacity, weight=1)
    
    # OPTIMIZATION: Pre-calculate all paths using multi-source Dijkstra
    log_print("Pre-calculating shortest paths from sinks using multi-source Dijkstra...")
    
    # Create data structures to store paths and distances
    all_paths = {}  # {sink_node: {source_node: path}}
    all_distances = {}  # {sink_node: {source_node: distance}}
    
    # Extract source and sink node IDs
    source_node_ids = [node_id for node_id, _, _ in source_nodes]
    
    # Pre-calculate for each sink using single_source_dijkstra
    path_failures = 0
    for sink_idx, (sink_node, sink_zip, _) in enumerate(sink_nodes):
        log_print(f"Calculating paths from sink {sink_idx+1}/{len(sink_nodes)}: {sink_zip}")
        
        try:
            # Calculate shortest paths from this sink to all nodes
            # Note: We use the sink as the source for Dijkstra, but paths will be reversed later
            length, paths = nx.single_source_dijkstra(G, sink_node, weight="weight")
            
            all_paths[sink_node] = {}
            all_distances[sink_node] = {}
            
            # Store paths to source nodes only (to save memory)
            for source_node in source_node_ids:
                if (source_node in paths):
                    # Reverse the path (it's from sink to source, we need source to sink)
                    all_paths[sink_node][source_node] = list(reversed(paths[source_node]))
                    all_distances[sink_node][source_node] = length[source_node]
                else:
                    path_failures += 1
        except Exception as e:
            log_print(f"Error calculating paths from sink {sink_zip}: {str(e)}")
            path_failures += len(source_node_ids)
    
    if path_failures > 0:
        log_print(f"Warning: Could not calculate {path_failures} paths. Will use high-cost alternatives.")
    
    # Now connect sources to sinks using the pre-calculated paths
    log_print("Creating bipartite graph edges using pre-calculated paths...")
    for i, (source_node, pop, source_zip) in enumerate(source_nodes):
        if i % 10 == 0:
            log_print(f"Processing source {i+1}/{len(source_nodes)}: {source_zip}")
        
        source_paths_found = 0
        for sink_node, sink_zip, sink_capacity in sink_nodes:
            # Check if we have a path from this source to this sink
            if sink_node in all_paths and source_node in all_paths[sink_node]:
                path = all_paths[sink_node][source_node]
                distance = all_distances[sink_node][source_node]
                
                # Calculate path properties
                min_capacity = float('inf')
                road_types = set()
                
                for u, v in zip(path[:-1], path[1:]):
                    segment_capacity = G[u][v].get('capacity', 10000)
                    road_type = G[u][v].get('road_type', 'UNKNOWN')
                    
                    min_capacity = min(min_capacity, segment_capacity)
                    road_types.add(road_type)
                
                # Add direct edge to bipartite graph
                edge_capacity = min(pop, sink_capacity, min_capacity)
                cost = max(1, min(999999, int(distance)))
                
                F.add_edge(source_node, sink_node, 
                          capacity=edge_capacity,
                          weight=cost, 
                          distance=distance,
                          road_types=list(road_types),
                          path=path)  # Store path for later use
                
                source_paths_found += 1
            else:
                # Add high-cost edge to ensure problem is feasible
                F.add_edge(source_node, sink_node, 
                          capacity=pop,
                          weight=999999,  # Very high cost = least desirable
                          distance=999999,
                          road_types=["NONE"])
        
        if source_paths_found == 0:
            log_print(f"Warning: No paths found for source {source_zip}. Using high-cost alternatives.")
    
    # Release memory from path calculation
    all_paths = None
    all_distances = None
    
    # Rest of the function remains the same...
    # ... (solver setup, solving, and results processing)
    log_print("Setting up min-cost flow solver...")
    mcf = min_cost_flow.SimpleMinCostFlow()
    
    # Create node maps for OR-Tools
    node_map = {}
    reverse_map = {}
    
    for i, node in enumerate(F.nodes()):
        node_map[node] = i
        reverse_map[i] = node
    
    # Add nodes with supplies/demands
    for node, data in F.nodes(data=True):
        supply = -data.get('demand', 0)  # Note: negative demand = positive supply
        if supply != 0:
            mcf.set_node_supply(node_map[node], int(supply))
    
    # Add edges as arcs
    edge_index = {}
    i = 0
    
    for u, v, data in F.edges(data=True):
        weight = int(data.get('weight', 1000))
        capacity = int(data.get('capacity', 1000))
        
        mcf.add_arc_with_capacity_and_unit_cost(
            node_map[u], node_map[v], capacity, weight)
        edge_index[(u, v)] = i
        i += 1
    
    # Check balance
    total_supply = 0
    total_demand = 0
    for node in range(mcf.num_nodes()):
        supply = mcf.supply(node)
        if supply > 0:
            total_supply += supply
        elif supply < 0:
            total_demand += -supply
            
    log_print(f"Solver balance check: Supply={total_supply}, Demand={total_demand}, Difference={total_supply-total_demand}")
    
    # Step 4: Solve the min-cost flow problem
    log_print(f"Solving min-cost flow with {mcf.num_nodes()} nodes and {mcf.num_arcs()} arcs...")
    status = mcf.solve()
    
    # Step 5: Process results
    result_data = {
        "centroid_flows": {},
        "sink_flows": {},
        "status": str(status),
        "optimal": (status == mcf.OPTIMAL)
    }
    
    # Initialize data structures for storing results
    for _, _, zip_code in source_nodes:
        result_data["centroid_flows"][zip_code] = {
            "population": 0,
            "sinks": {}
        }
    
    for _, sink_zip, _ in sink_nodes:
        result_data["sink_flows"][sink_zip] = {
            "total_inflow": 0,
            "centroids": {}
        }
    
    # Process results based on solver status
    if status == mcf.OPTIMAL:
        log_print("Min-cost flow solver found optimal solution!")
        
        # Extract direct flows from centroids to sinks
        flow_count = 0
        for source_node, pop, source_zip in source_nodes:
            result_data["centroid_flows"][source_zip]["population"] = pop
            
            # Get all outgoing flows from this source
            source_total_flow = 0
            for sink_node, sink_zip, _ in sink_nodes:
                if (source_node, sink_node) in edge_index:
                    idx = edge_index[(source_node, sink_node)]
                    flow = mcf.flow(idx)
                    
                    if flow > 0:
                        flow_count += 1
                        source_total_flow += flow
                        
                        # Record flow in results
                        if sink_zip not in result_data["centroid_flows"][source_zip]["sinks"]:
                            result_data["centroid_flows"][source_zip]["sinks"][sink_zip] = 0
                        result_data["centroid_flows"][source_zip]["sinks"][sink_zip] += flow
                        
                        # Also record in sink data
                        result_data["sink_flows"][sink_zip]["total_inflow"] += flow
                        if source_zip not in result_data["sink_flows"][sink_zip]["centroids"]:
                            result_data["sink_flows"][sink_zip]["centroids"][source_zip] = 0
                        result_data["sink_flows"][sink_zip]["centroids"][source_zip] += flow
                        
                        # Get the path details
                        distance = F[source_node][sink_node].get('distance', 0)
                        road_types = F[source_node][sink_node].get('road_types', [])
                        
                        log_print(f"  Flow: {source_zip} → {sink_zip}: {flow:,} people ({distance/1000:.1f} km)")
            
            # Verify total flow matches population
            if abs(source_total_flow - pop) > 0.1:
                log_print(f"  Warning: Source {source_zip} has flow discrepancy: {pop} population but {source_total_flow} total flow")
        
        # Verify sink flows
        for _, sink_zip, capacity in sink_nodes:
            sink_flow = result_data["sink_flows"][sink_zip]["total_inflow"]
            if sink_flow > capacity + 0.1:  # Small tolerance for floating point errors
                log_print(f"  Warning: Sink {sink_zip} received {sink_flow} flow, exceeding capacity {capacity}")
        
        log_print(f"Total flow assignments: {flow_count}")
                    
    else:
        log_print(f"Min-cost flow solver status: {status} - Not optimal. Using distance-based allocation.")
        
        # Fallback to distance-based allocation
        for node, pop, centroid_zip in source_nodes:
            result_data["centroid_flows"][centroid_zip]["population"] = pop
            
            # Find paths to all sinks
            paths_to_sinks = []
            for sink_node, sink_zip, _ in sink_nodes:
                try:
                    path = nx.shortest_path(G, node, sink_node, weight="weight")
                    distance = sum(G[u][v].get('weight', 1) for u, v in zip(path[:-1], path[1:]))
                    
                    paths_to_sinks.append({
                        "sink": sink_zip,
                        "sink_node": sink_node,
                        "distance": distance,
                        "path": path,
                        "weight": 1/max(0.1, distance**2)  # Inverse squared distance
                    })
                except nx.NetworkXNoPath:
                    continue
            
            if not paths_to_sinks:
                log_print(f"  WARNING: No paths found from {centroid_zip} to any sink!")
                continue
            
            # Use the 5 closest sinks
            closest_sinks = sorted(paths_to_sinks, key=lambda x: x["distance"])[:min(5, len(paths_to_sinks))]
            
            # Calculate total weight
            total_weight = sum(x["weight"] for x in closest_sinks)
            
            # Distribute population based on weights
            for path_data in closest_sinks:
                sink_zip = path_data["sink"]
                weight = path_data["weight"]
                proportion = weight / total_weight if total_weight > 0 else 0
                flow = pop * proportion
                
                # Record flow
                if sink_zip not in result_data["centroid_flows"][centroid_zip]["sinks"]:
                    result_data["centroid_flows"][centroid_zip]["sinks"][sink_zip] = 0
                result_data["centroid_flows"][centroid_zip]["sinks"][sink_zip] += flow
                
                # Also record in sink data
                result_data["sink_flows"][sink_zip]["total_inflow"] += flow
                if centroid_zip not in result_data["sink_flows"][sink_zip]["centroids"]:
                    result_data["sink_flows"][sink_zip]["centroids"][centroid_zip] = 0
                result_data["sink_flows"][sink_zip]["centroids"][centroid_zip] += flow
    
    # Generate report
    report_path = os.path.join(output_dir, "min_cost_flow_report.txt")
    with open(report_path, 'w') as f:
        f.write("MIN-COST FLOW EVACUATION ANALYSIS\n")
        f.write("================================\n\n")
        
        f.write(f"Total Population: {total_population:,}\n")
        f.write(f"Solver Status: {status}\n")
        f.write(f"Solution: {'OPTIMAL' if status == mcf.OPTIMAL else 'NON-OPTIMAL'}\n\n")
        
        # Write sink summary
        f.write("SINK DISTRIBUTION SUMMARY\n")
        f.write("-----------------------\n")
        
        for sink_zip, data in sorted(result_data["sink_flows"].items(), key=lambda x: x[1]["total_inflow"], reverse=True):
            inflow = data["total_inflow"]
            percentage = (inflow / total_population) * 100 if total_population > 0 else 0
            f.write(f"Sink {sink_zip}: {inflow:,.0f} people ({percentage:.1f}% of total)\n")
        
        f.write("\n\nDETAILED FLOW BY CENTROID\n")
        f.write("========================\n\n")
        
        # Write detailed flow for each centroid
        for centroid_zip, data in sorted(result_data["centroid_flows"].items(), key=lambda x: x[1]["population"], reverse=True):
            population = data["population"]
            sinks_data = data.get("sinks", {})
            
            f.write(f"\nZIP Code: {centroid_zip} (Population: {population:,})\n")
            f.write("-" * 40 + "\n")
            
            if sinks_data:
                sink_flows = [(sink_zip, flow) for sink_zip, flow in sinks_data.items()]
                sink_flows.sort(key=lambda x: x[1], reverse=True)
                
                for sink_zip, flow in sink_flows:
                    portion = flow / population if population > 0 else 0
                    if flow > 0:
                        f.write(f"  → {sink_zip}: {flow:,.0f} people ({portion*100:.1f}%)\n")
            else:
                f.write("  No flow data available\n")
    
    # LEVEL 2: Generate detailed flow paths by mapping flows back to full network
    if status == mcf.OPTIMAL:
        log_print("Generating detailed evacuation routes...")
        route_data = {}
        
        for source_node, _, source_zip in source_nodes:
            routes_for_source = []
            
            for sink_node, sink_zip, _ in sink_nodes:
                if (source_node, sink_node) in edge_index:
                    idx = edge_index[(source_node, sink_node)]
                    flow = mcf.flow(idx)
                    
                    if flow > 0:
                        try:
                            # Find the actual path through the full network
                            path = nx.shortest_path(G, source_node, sink_node, weight="weight")
                            path_distance = sum(G[u][v].get('weight', 1) for u, v in zip(path[:-1], path[1:]))
                            
                            # Extract road types along the path
                            road_segments = []
                            for u, v in zip(path[:-1], path[1:]):
                                road_type = G[u][v].get('road_type', 'UNKNOWN')
                                segment_length = G[u][v].get('weight', 0)
                                road_segments.append({
                                    "from_node": u,
                                    "to_node": v,
                                    "road_type": road_type,
                                    "length": segment_length
                                })
                            
                            # Store the detailed route data
                            routes_for_source.append({
                                "sink": sink_zip,
                                "flow": flow,
                                "distance": path_distance,
                                "path_nodes": path,
                                "road_segments": road_segments
                            })
                            
                        except nx.NetworkXNoPath:
                            log_print(f"Warning: Could not find actual path from {source_zip} to {sink_zip}")
            
            route_data[source_zip] = routes_for_source
        
        # Add route data to result
        result_data["routes"] = route_data
    
    # Save result data
    json_path = os.path.join(output_dir, "min_cost_flow_results.json")
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    # Log completion
    calculation_time = time.time() - start_time
    log_print(f"Min-cost flow calculation completed in {calculation_time:.2f} seconds")
    log_print(f"Results saved to {report_path} and {json_path}")
    
    return report_path, json_path

def extract_max_evacuation_time(report_path):
    """Extract the maximum evacuation time from a report file."""
    max_time = 0
    with open(report_path, 'r') as f:
        for line in f:
            if "Total evacuation time:" in line:
                try:
                    time_str = line.split("Total evacuation time:")[1].split("hours")[0].strip()
                    time_val = float(time_str)
                    max_time = max(max_time, time_val)
                except:
                    pass
    return max_time

def calculate_evacuation_flow_report(G, output_dir, congestion_scenario='moderate_congestion', phases=12):
    """Calculate evacuation flow and time with congestion factors."""
    log_print(f"Calculating evacuation flow and time for {congestion_scenario} scenario...")
    
    # Look up the congestion factor
    congestion_scenarios = {
        'free_flow': 0.7,           # Minimal evacuation traffic
        'light_congestion': 0.5,    # Light evacuation traffic
        'moderate_congestion': 0.3, # Typical evacuation conditions
        'heavy_congestion': 0.2,    # Heavy evacuation traffic
        'severe_congestion': 0.1    # Severe congestion with bottlenecks
    }
    congestion_factor = congestion_scenarios.get(congestion_scenario, 0.3)
    
    # First calculate the optimal evacuation flows
    scenario_dir = os.path.join(output_dir, congestion_scenario)
    os.makedirs(scenario_dir, exist_ok=True)
    
    report_path, json_path = calculate_min_cost_flow(G, scenario_dir)
    
    # Load the flow results
    with open(json_path, 'r') as f:
        flow_data = json.load(f)
    
    # Road properties for speed calculation
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
    
    # Calculate evacuation times for each route
    if flow_data.get("optimal", False) and "routes" in flow_data:
        log_print("Calculating evacuation times based on flow results...")
        total_population = 0
        max_evacuation_time = 0
        total_flow_volume = 0  # Track total flow for network congestion calculation
        
        # Track road usage for bottleneck calculations
        road_usage = {}  # Will store {road_segment_id: total_flow}
        
        # First pass - calculate basic travel times and track road segment usage
        for source_zip, routes in flow_data["routes"].items():
            population = flow_data["centroid_flows"][source_zip]["population"]
            total_population += population
            
            for route in routes:
                flow = route["flow"]
                total_flow_volume += flow
                
                # Track which road segments are used by this flow
                for segment in route["road_segments"]:
                    segment_id = f"{segment['from_node']}_{segment['to_node']}"
                    if segment_id not in road_usage:
                        road_usage[segment_id] = 0
                    road_usage[segment_id] += flow
        
        # Calculate average flow per segment for bottleneck scaling
        avg_flow_per_segment = total_flow_volume / max(1, len(road_usage))
        
        # Second pass - calculate actual evacuation times with bottlenecks
        for source_zip, routes in flow_data["routes"].items():
            zip_max_time = 0  # Track maximum evacuation time for this zip code
            
            for route in routes:
                sink_zip = route["sink"]
                flow = route["flow"]
                distance = route["distance"]  # in meters
                
                # Calculate travel time based on road segments, speeds, and congestion
                route_time = 0
                for segment in route["road_segments"]:
                    road_type = segment["road_type"]
                    segment_length = segment["length"]  # in meters
                    segment_id = f"{segment['from_node']}_{segment['to_node']}"
                    
                    # Get base speed for this road type (mph)
                    if len(road_type) == 1:
                        base_speed = road_properties.get(road_type, {}).get('speed', 30)
                    else:
                        base_speed = road_properties.get(road_type, {}).get('speed', 30)
                    
                    # Apply congestion factor to base speed
                    speed_mph = base_speed * congestion_factor
                    
                    # Additional congestion based on segment usage
                    segment_flow = road_usage.get(segment_id, 0)
                    segment_usage_ratio = segment_flow / avg_flow_per_segment if avg_flow_per_segment > 0 else 1
                    
                    # More traffic on a segment means lower speeds
                    local_congestion_factor = max(0.3, 1.0 - (0.5 * min(1.0, segment_usage_ratio)))
                    speed_mph *= local_congestion_factor
                    
                    # Convert mph to m/s (1 mph = 0.44704 m/s)
                    speed_ms = speed_mph * 0.44704
                    
                    # Calculate time for this segment in hours
                    segment_time_hours = segment_length / (speed_ms * 3600)
                    route_time += segment_time_hours
                
                # Calculate bottleneck effect (delay increases with flow volume)
                # More people = more vehicles = more congestion at bottlenecks
                vehicles_per_hour = 2000  # Assume 2000 vehicles/hour can exit an area
                capacity_ratio = flow / (vehicles_per_hour * 1.5)  # 1.5 people per vehicle average
                bottleneck_hours = max(0, capacity_ratio - 1)  # Only add when over capacity
                
                # Apply phased evacuation model
                # People don't all leave at once - waves of departures
                wave_size = 2000  # People per evacuation wave
                num_waves = max(1, int(flow / wave_size))
                wave_spacing_hours = 1.0  # Time between waves
                phased_evacuation_hours = (num_waves - 1) * wave_spacing_hours
                
                # Combine all time components for total evacuation time
                total_route_time = route_time + bottleneck_hours + phased_evacuation_hours
                
                # Apply regional effect - evacuations slow down as more of the system is used
                regional_congestion_multiplier = 1.0 + (0.2 * (total_population / 500000))
                scaled_route_time = total_route_time * regional_congestion_multiplier
                
                log_print(f"  {source_zip} → {sink_zip}: {flow:,} people, {distance/1000:.1f} km")
                log_print(f"    Base travel time: {route_time:.2f} hrs, Bottleneck: {bottleneck_hours:.2f} hrs")
                log_print(f"    Phased departure: {phased_evacuation_hours:.2f} hrs ({num_waves} waves)")
                log_print(f"    Total evacuation: {scaled_route_time:.2f} hrs")
                
                # Store the evacuation time in the route data
                route["travel_time_hours"] = route_time
                route["bottleneck_hours"] = bottleneck_hours
                route["phased_evacuation_hours"] = phased_evacuation_hours
                route["total_evacuation_time_hours"] = scaled_route_time
                
                # Update maximum times
                zip_max_time = max(zip_max_time, scaled_route_time)
                max_evacuation_time = max(max_evacuation_time, scaled_route_time)
            
            # Store the maximum time for this zip code
            flow_data["centroid_flows"][source_zip]["max_evacuation_time"] = zip_max_time
        
        # Apply scenario-specific scaling factor
        scenario_multiplier = {
            'free_flow': 1.0,
            'light_congestion': 1.2,
            'moderate_congestion': 1.5,
            'heavy_congestion': 2.0,
            'severe_congestion': 3.0
        }.get(congestion_scenario, 1.5)
        
        # Calculate final evacuation time
        final_max_evacuation_time = max_evacuation_time * scenario_multiplier
        
        log_print(f"Maximum base evacuation time: {max_evacuation_time:.2f} hours")
        log_print(f"Scenario multiplier: {scenario_multiplier:.1f}x")
        log_print(f"TOTAL EVACUATION TIME: {final_max_evacuation_time:.2f} hours")
        
        # Update the JSON data with evacuation time information
        flow_data["evacuation_parameters"] = {
            "congestion_scenario": congestion_scenario,
            "congestion_factor": congestion_factor,
            "evacuation_phases": phases,
            "max_evacuation_time_hours": final_max_evacuation_time,
            "base_evacuation_time_hours": max_evacuation_time,
            "scenario_multiplier": scenario_multiplier
        }
        
        # Save updated JSON
        with open(json_path, 'w') as f:
            json.dump(flow_data, f, indent=2)
            
        # Update the report with evacuation time
        time_report_path = os.path.join(scenario_dir, "evacuation_time_report.txt")
        with open(time_report_path, 'w') as f:
            f.write(f"EVACUATION TIME REPORT: {congestion_scenario.upper()}\n")
            f.write("==========================================\n\n")
            f.write(f"Congestion Factor: {congestion_factor}\n")
            f.write(f"Total Population: {total_population:,}\n")
            f.write(f"Evacuation Phases: {phases}\n")
            f.write(f"Total evacuation time: {final_max_evacuation_time:.2f} hours\n\n")
            
            f.write("EVACUATION TIME BREAKDOWN\n")
            f.write("------------------------\n")
            f.write(f"Base maximum travel time: {max_evacuation_time:.2f} hours\n")
            f.write(f"Scenario multiplier: {scenario_multiplier:.1f}x\n\n")
            
            f.write("EVACUATION TIMES BY ZIP CODE\n")
            f.write("---------------------------\n\n")
            
            # Show evacuation time for each zip code
            for source_zip, data in sorted(flow_data["centroid_flows"].items()):
                population = data["population"]
                evac_time = data.get("max_evacuation_time", 0) * scenario_multiplier
                
                f.write(f"{source_zip} (Pop: {population:,}): {evac_time:.2f} hours\n")
        
        # Return the evacuation time report path
        return time_report_path
    else:
        log_print("Could not calculate evacuation times: No optimal route data available")
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
    
    # Load or create graph
    if os.path.exists(graph_pickle_path):
        G = load_graph_from_pickle(graph_pickle_path)
    else:
        routes_gdf, centroids_gdf, sinks_gdf = load_data(routes_path, centroids_path, sink_nodes_path)
        G = create_connected_graph(routes_gdf, centroids_gdf, sinks_gdf, connection_threshold=150)
        G = connect_sinks_to_terminals(G, terminal_nodes_path, sink_nodes_path)
        save_graph_to_pickle(G, graph_pickle_path)
    
    # Assign capacities and population data to the graph
    G = assign_road_capacities_and_population(G, population_data_path, sink_nodes_path)
    
    # Run analysis with multiple congestion scenarios
    congestion_scenarios = {
        'free_flow': 0.7,           # Minimal evacuation traffic
        'light_congestion': 0.5,    # Light evacuation traffic
        'moderate_congestion': 0.3, # Typical evacuation conditions
        'heavy_congestion': 0.2,    # Heavy evacuation traffic
        'severe_congestion': 0.1    # Severe congestion with bottlenecks
    }
    
    log_print("\n===== COMPARING CONGESTION SCENARIOS =====")
    log_print("Scenario            | Congestion Factor | Evacuation Time (hours)")
    log_print("-------------------|-------------------|---------------------")
    
    scenario_results = {}
    for scenario, factor in congestion_scenarios.items():
        log_print(f"Running scenario: {scenario} (factor: {factor})...")
        report_path = calculate_evacuation_flow_report(G, results_dir, 
                                                      congestion_scenario=scenario,
                                                      phases=12)
        
        # Extract evacuation time from the report (this would need to be implemented)
        max_evacuation_time = extract_max_evacuation_time(report_path)
        scenario_results[scenario] = max_evacuation_time
        
        log_print(f"{scenario.ljust(20)} | {factor:.1f}                | {max_evacuation_time:.2f}")
    
    # Generate comparative report
    comparative_report_path = os.path.join(results_dir, "congestion_comparison.txt")
    with open(comparative_report_path, 'w') as f:
        f.write("TAMPA BAY EVACUATION TIME COMPARISON BY CONGESTION SCENARIO\n")
        f.write("=========================================================\n\n")
        
        for scenario, evac_time in sorted(scenario_results.items(), key=lambda x: x[1]):
            f.write(f"{scenario.ljust(20)}: {evac_time:.2f} hours\n")
    
    total_time = time.time() - total_start_time
    log_print(f"\nAll scenarios completed in {total_time:.2f} seconds")
    log_print(f"Comparative report saved to: {comparative_report_path}")

if __name__ == "__main__":
    main()