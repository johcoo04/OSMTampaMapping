import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import os
import pickle
from shapely.geometry import Point, LineString
import time
import json
import pandas as pd

def load_data(routes_path, intersections_path):
    routes_gdf = gpd.read_file(routes_path)
    intersections_gdf = gpd.read_file(intersections_path)
    
    # Ensure both have the same CRS (coordinate reference system)
    routes_gdf = routes_gdf.to_crs(epsg=3857)
    intersections_gdf = intersections_gdf.to_crs(epsg=3857)
    
    print(f"Loaded {len(routes_gdf)} evacuation routes")
    print(f"Loaded {len(intersections_gdf)} intersections")
    
    return routes_gdf, intersections_gdf

def build_topological_graph(routes_gdf, intersections_gdf):
    print("Building graph with topological connections...")
    start_time = time.time()
    
    # Initialize new graph
    G = nx.Graph()
    
    for idx, row in intersections_gdf.iterrows():
        G.add_node(f"i{idx}", 
                   pos=(row.geometry.x, row.geometry.y), 
                   node_type='intersection',
                   geometry=row.geometry)
    
    endpoint_nodes = {}  # Maps (x,y) coordinates to node IDs
    
    for idx, route in routes_gdf.iterrows():
        if route.geometry.geom_type != 'LineString':
            continue
        
        road_name = route.get('STREET', 'unnamed')
        road_type = route.get('TYPE', 'U')
        coords = list(route.geometry.coords)
        
        # Get start and end points
        start_point = coords[0]
        end_point = coords[-1]
        
        # Create or reuse nodes for the endpoints
        start_node = f"r{idx}_start"
        end_node = f"r{idx}_end"
        
        # Add nodes if they don't exist
        G.add_node(start_node, pos=start_point, node_type='route_endpoint')
        G.add_node(end_node, pos=end_point, node_type='route_endpoint')
        
        # Add the route as an edge
        G.add_edge(start_node, end_node, 
                  road_name=road_name,
                  road_type=road_type, 
                  weight=route.geometry.length,
                  geometry=route.geometry)
        
        # Store endpoints for connection finding
        endpoint_nodes[start_point] = start_node
        endpoint_nodes[end_point] = end_node
    
    print(f"Added {len(routes_gdf)} route edges to the graph")
    
    # Connect route endpoints to nearby intersections
    print("Connecting routes to intersections...")
    connections = 0
    
    # For each route endpoint, find if it's near an intersection
    for coord, node_id in endpoint_nodes.items():
        point = Point(coord)
        
        # Find intersections within a small distance (adjust tolerance as needed)
        nearby = intersections_gdf[intersections_gdf.geometry.distance(point) < 0.001]
        
        if not nearby.empty:
            # Connect to the closest intersection
            closest_idx = nearby.distance(point).idxmin()
            intersection_node = f"i{closest_idx}"
            
            # Only add connection if it doesn't exist
            if not G.has_edge(node_id, intersection_node):
                G.add_edge(node_id, intersection_node, 
                          road_name="connector", 
                          road_type="C", 
                          weight=point.distance(nearby.loc[closest_idx].geometry))
                connections += 1
    
    print(f"Added {connections} connections between routes and intersections")
    
    # Remove isolated nodes
    isolated = [node for node, degree in G.degree() if degree == 0]
    G.remove_nodes_from(isolated)
    print(f"Removed {len(isolated)} isolated nodes")
    
    # Print final graph statistics
    total_time = time.time() - start_time
    print(f"\nTotal graph creation time: {total_time:.4f} seconds")
    print(f"Final graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G

def add_centroids_to_graph(G, centroids_gdf, routes_gdf):
    """
    Add centroids to the graph and connect them to the nearest road endpoint (start or end)
    
    Parameters:
    -----------
    G : networkx.Graph
        The existing road network graph
    centroids_gdf : GeoDataFrame
        GeoDataFrame with centroid points (zip code centroids)
    routes_gdf : GeoDataFrame
        Original routes GeoDataFrame used for finding connections
    """
    print("Adding centroids to the graph...")
    start_time = time.time()
    
    centroid_connections = 0
    
    # First add all centroids as nodes
    for idx, row in centroids_gdf.iterrows():
        # Create a unique ID for each centroid
        centroid_id = f"c{idx}"
        
        # Add the centroid as a node
        G.add_node(centroid_id, 
                 pos=(row.geometry.x, row.geometry.y),
                 node_type='centroid',
                 geometry=row.geometry,
                 zip_code=row.get('ZIP_CODE', ''),     # Add zip code if available
                 name=row.get('NAME', f'Centroid {idx}'))  # Add name if available
    
    print(f"Added {len(centroids_gdf)} centroids as nodes")
    
    # Connect each centroid to the nearest endpoint of the nearest road
    for idx, centroid in centroids_gdf.iterrows():
        centroid_id = f"c{idx}"
        centroid_point = centroid.geometry
        
        # Find the nearest road segments
        distances = routes_gdf.geometry.apply(lambda g: g.distance(centroid_point))
        nearest_route_idx = distances.idxmin()  # Get the closest road
        nearest_route = routes_gdf.loc[nearest_route_idx]
        
        # Get both start and end nodes from G
        start_node = f"r{nearest_route_idx}_start"
        end_node = f"r{nearest_route_idx}_end"
        
        # Calculate distance to both endpoints and connect to the closer one
        connections_added = 0
        
        start_dist = float('inf')
        end_dist = float('inf')
        
        # Check and calculate distance to start node
        if start_node in G.nodes:
            start_pos = G.nodes[start_node]['pos']
            start_dist = Point(start_pos).distance(centroid_point)
        
        # Check and calculate distance to end node
        if end_node in G.nodes:
            end_pos = G.nodes[end_node]['pos']
            end_dist = Point(end_pos).distance(centroid_point)
        
        # Connect to the closest endpoint
        if start_dist <= end_dist and start_node in G.nodes:
            # Start node is closer
            G.add_edge(centroid_id, start_node,
                      road_name="centroid_connector",
                      road_type="Z",  # Z for centroid connector
                      weight=start_dist)
            connections_added += 1
        elif end_dist < start_dist and end_node in G.nodes:
            # End node is closer
            G.add_edge(centroid_id, end_node,
                      road_name="centroid_connector",
                      road_type="Z",  # Z for centroid connector
                      weight=end_dist)
            connections_added += 1
        
        # If no connections were made, try with the second nearest road
        if connections_added == 0:
            print(f"Warning: Could not connect centroid {idx} (ZIP {centroid.get('ZIP_CODE', '')}) to nearest road")
            
            # Try second-closest road as fallback
            if len(distances) > 1:
                second_nearest = distances.nsmallest(2).index[1]
                start_node_2 = f"r{second_nearest}_start"
                end_node_2 = f"r{second_nearest}_end"
                
                # Calculate distances to both endpoints of second nearest road
                start_dist_2 = float('inf')
                end_dist_2 = float('inf')
                
                if start_node_2 in G.nodes:
                    start_pos_2 = G.nodes[start_node_2]['pos']
                    start_dist_2 = Point(start_pos_2).distance(centroid_point)
                
                if end_node_2 in G.nodes:
                    end_pos_2 = G.nodes[end_node_2]['pos']
                    end_dist_2 = Point(end_pos_2).distance(centroid_point)
                
                # Connect to the closest endpoint of second nearest road
                if start_dist_2 <= end_dist_2 and start_node_2 in G.nodes:
                    G.add_edge(centroid_id, start_node_2,
                              road_name="centroid_connector",
                              road_type="Z",
                              weight=start_dist_2)
                    connections_added += 1
                elif end_dist_2 < start_dist_2 and end_node_2 in G.nodes:
                    G.add_edge(centroid_id, end_node_2,
                              road_name="centroid_connector",
                              road_type="Z",
                              weight=end_dist_2)
                    connections_added += 1
        
        centroid_connections += connections_added
    
    print(f"Added {centroid_connections} connections from centroids to roads")
    
    total_time = time.time() - start_time
    print(f"Centroids added in {total_time:.4f} seconds")
    return G

def visualize_graph(G, routes_gdf, intersections_gdf, output_path=None):
    """Visualize the graph with centroids as dots and sink nodes as stars"""
    start_time = time.time()
    print("Starting visualization with matplotlib...")
    
    fig, ax = plt.subplots(figsize=(45, 45))
    
    # Extract node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Define colors for different road types
    road_colors = {
        'C': 'blue',     # County road
        'H': 'red',      # Highway
        'M': 'green',    # Main road
        'P': 'purple',   # Primary/Peripheral road
        'Z': 'magenta',  # Centroid connector
        'S': 'orange',   # Sink connector
        'E': 'yellow',   # Evacuation connector (new)
        'U': 'gray'      # Unknown type
    }
    
    # Draw edges with appropriate colors
    edge_types_drawn = set()
    
    for u, v, data in G.edges(data=True):
        if u in pos and v in pos:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            road_type = data.get('road_type', 'U')
            # Skip certain road types if needed
            if road_type in ['I', 'J']:
                continue
                
            color = road_colors.get(road_type, 'gray')
            
            # Draw with appropriate width and style
            linewidth = 2.0
            alpha = 0.8
            
            # Make connectors thinner
            if road_type in ['Z', 'S']:
                linewidth = 1.0
                alpha = 0.6
            
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, 
                   alpha=alpha, label=f'Type {road_type}' if road_type not in edge_types_drawn else "")
            
            edge_types_drawn.add(road_type)
    
    # Draw centroids as dots
    centroid_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'centroid']
    if centroid_nodes:
        centroid_scatter = ax.scatter(
            [pos[n][0] for n in centroid_nodes if n in pos],
            [pos[n][1] for n in centroid_nodes if n in pos],
            s=100, c='blue', alpha=0.8, marker='o', edgecolor='black', label='Population Centers'
        )
        
        # Removed: ZIP code labels for centroids
    
    # Draw sink nodes as stars
    sink_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'sink']
    if sink_nodes:
        sink_scatter = ax.scatter(
            [pos[n][0] for n in sink_nodes if n in pos],
            [pos[n][1] for n in sink_nodes if n in pos],
            s=200, c='yellow', alpha=1.0, marker='*', edgecolor='black', label='Evacuation Points'
        )
        
        # Removed: ZIP code labels for sink nodes
    
    # Add legend with road types and node types
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
    
    # Set title and labels
    ax.set_title('Tampa Evacuation Network with Population Centers and Evacuation Points')
    
    # Remove axes
    ax.set_axis_off()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to {output_path}")
    
    # No plt.show() call
    
    total_time = time.time() - start_time
    print(f"Visualization completed in {total_time:.4f} seconds")

def visualize_with_perimeter(G, perimeter_nodes, routes_gdf, intersections_gdf, output_path=None):
    """Visualize the graph with perimeter nodes highlighted"""
    start_time = time.time()
    print("Starting visualization with perimeter nodes...")
    
    fig, ax = plt.subplots(figsize=(45, 45))
    
    # Extract node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Define colors for different road types
    road_colors = {
        'C': 'blue',     # County road
        'H': 'red',      # Highway
        'M': 'green',    # Main road
        'P': 'purple',   # Primary/Peripheral road
        'Z': 'magenta',  # Centroid connector
        'S': 'orange',   # Sink connector
        'E': 'brown',    # Evacuation connector
        'U': 'gray'      # Unknown type
    }
    
    # Draw edges with appropriate colors
    edge_types_drawn = set()
    
    for u, v, data in G.edges(data=True):
        if u in pos and v in pos:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            road_type = data.get('road_type', 'U')
            # Skip certain road types if needed
            if road_type in ['I', 'J']:
                continue
                
            color = road_colors.get(road_type, 'gray')
            
            # Draw with appropriate width and style
            linewidth = 2.0
            alpha = 0.8
            
            # Make connectors thinner
            if road_type in ['Z', 'S', 'E']:
                linewidth = 1.0
                alpha = 0.6
            
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, 
                   alpha=alpha, label=f'Type {road_type}' if road_type not in edge_types_drawn else "")
            
            edge_types_drawn.add(road_type)
    
    # Draw centroids as dots
    centroid_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'centroid']
    if centroid_nodes:
        centroid_scatter = ax.scatter(
            [pos[n][0] for n in centroid_nodes if n in pos],
            [pos[n][1] for n in centroid_nodes if n in pos],
            s=100, c='blue', alpha=0.8, marker='o', edgecolor='black', label='Population Centers'
        )
    
    # Draw sink nodes as stars
    sink_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'sink']
    if sink_nodes:
        sink_scatter = ax.scatter(
            [pos[n][0] for n in sink_nodes if n in pos],
            [pos[n][1] for n in sink_nodes if n in pos],
            s=200, c='yellow', alpha=1.0, marker='*', edgecolor='black', label='Evacuation Points'
        )
    
    # Highlight perimeter nodes
    perimeter_positions = [pos[n] for n in perimeter_nodes if n in pos]
    if perimeter_positions:
        x_coords = [p[0] for p in perimeter_positions]
        y_coords = [p[1] for p in perimeter_positions]
        perimeter_scatter = ax.scatter(
            x_coords, y_coords, 
            s=150, c='red', marker='D', edgecolor='black', linewidth=1, label='Perimeter Nodes'
        )
    
    # Add legend with road types and node types
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
    
    # Set title and labels
    ax.set_title('Tampa Evacuation Network with Perimeter Nodes Highlighted')
    
    # Remove axes
    ax.set_axis_off()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to {output_path}")
    
    total_time = time.time() - start_time
    print(f"Visualization with perimeter nodes completed in {total_time:.4f} seconds")

def load_centroids(centroids_path, target_crs=3857):
    """Load centroids from the specific zip code JSON format"""

    
    print(f"Loading centroids from: {centroids_path}")
    
    # Check if file exists
    if not os.path.exists(centroids_path):
        print(f"ERROR: File not found at {centroids_path}")
        raise FileNotFoundError(f"Centroids file not found: {centroids_path}")
    
    try:
        # Read the JSON file
        with open(centroids_path, 'r') as f:
            data = json.load(f)
        
        # Create centroids list from the ZIP code dictionary format
        centroids = []
        for zip_code, coords in data.items():
            if "centroid_x" in coords and "centroid_y" in coords:
                # Create a Point for each zip code
                point = Point(coords["centroid_x"], coords["centroid_y"])
                centroids.append({
                    "geometry": point,
                    "ZIP_CODE": zip_code,
                    "NAME": f"ZIP {zip_code}"
                })
        
        # Create GeoDataFrame from centroids
        if centroids:
            centroids_gdf = gpd.GeoDataFrame(centroids, crs=f"EPSG:{target_crs}")
            print(f"Successfully loaded {len(centroids_gdf)} zip code centroids")
            return centroids_gdf
        else:
            print("No valid centroids found in the JSON file")
            raise ValueError("No valid centroids found")
            
    except Exception as e:
        print(f"Error loading centroids: {str(e)}")
        raise

def load_sink_nodes(sink_nodes_path, target_crs=3857):
    """Load sink nodes (evacuation destinations) from JSON file"""
    print(f"Loading sink nodes from: {sink_nodes_path}")
    
    # Check if file exists
    if not os.path.exists(sink_nodes_path):
        print(f"ERROR: File not found at {sink_nodes_path}")
        raise FileNotFoundError(f"Sink nodes file not found: {sink_nodes_path}")
    
    try:
        # Read the JSON file
        with open(sink_nodes_path, 'r') as f:
            data = json.load(f)
        
        # Filter for only the specified ZIP codes
        keep_zips = ['34638', '33544', '33541', '33849', '33815', 
                     '33811', '33860', '33834', '334219', '334221']
        
        # Create sink nodes list from the ZIP code dictionary format
        sink_nodes = []
        for zip_code, coords in data.items():
            # Only include the specified ZIP codes
            if zip_code in keep_zips and "centroid_x" in coords and "centroid_y" in coords:
                # Create a Point for each sink node
                point = Point(coords["centroid_x"], coords["centroid_y"])
                sink_nodes.append({
                    "geometry": point,
                    "ZIP_CODE": zip_code,
                    "NAME": f"SINK {zip_code}"
                })
        
        # Create GeoDataFrame from sink nodes
        if sink_nodes:
            sink_nodes_gdf = gpd.GeoDataFrame(sink_nodes, crs=f"EPSG:{target_crs}")
            print(f"Successfully loaded {len(sink_nodes_gdf)} sink nodes")
            return sink_nodes_gdf
        else:
            print("No valid sink nodes found in the JSON file")
            raise ValueError("No valid sink nodes found")
            
    except Exception as e:
        print(f"Error loading sink nodes: {str(e)}")
        raise

def connect_dead_end_nodes_to_sinks(G, max_connections_per_terminal=2, max_distance=100000):
    """
    Connect only true dead-end nodes (isolated terminals with no continuation) to sink nodes.
    
    Parameters:
    -----------
    G : networkx.Graph
        The road network graph with sink nodes already added
    max_connections_per_terminal : int
        Maximum number of sinks to connect to each terminal node (default: 2)
    max_distance : float
        Maximum distance to consider for connections (default: 100000 units)
    
    Returns:
    --------
    G : networkx.Graph
        Updated graph with dead-end nodes connected to sink nodes
    """
    print(f"Identifying and connecting true dead-end nodes to sinks...")
    start_time = time.time()
    
    # Get all node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Get all sink nodes
    sink_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'sink']
    if not sink_nodes:
        print("No sink nodes found in the graph. Cannot connect terminal nodes.")
        return G
    
    print(f"Found {len(sink_nodes)} sink nodes for evacuation connections")
    
    # Find connected components in the original road network (excluding sink and centroid connectors)
    # This helps identify real road network structure
    road_network = G.copy()
    
    # Remove existing connections to sinks and centroids
    edges_to_remove = []
    for u, v, data in road_network.edges(data=True):
        road_type = data.get('road_type', '')
        if road_type in ['Z', 'S', 'E']:  # Centroid, Sink, or Evacuation connectors
            edges_to_remove.append((u, v))
    
    road_network.remove_edges_from(edges_to_remove)
    
    # Find true terminal nodes - actual dead ends in the road network
    true_terminals = []
    
    for node, degree in road_network.degree():
        # Look for nodes with degree=1 (only one connection) in the road network
        if degree == 1:
            node_type = G.nodes[node].get('node_type', '')
            
            # Make sure it's a route endpoint
            if node_type == 'route_endpoint' and node in pos:
                # Check if this terminal is already connected to any sink or centroid
                connected_to_special = False
                for neighbor in G.neighbors(node):
                    neighbor_type = G.nodes[neighbor].get('node_type', '')
                    if neighbor_type in ['sink', 'centroid']:
                        connected_to_special = True
                        break
                
                if not connected_to_special:
                    true_terminals.append(node)
    
    print(f"Found {len(true_terminals)} true dead-end nodes in the road network")
    
    # For each terminal, find and connect to the closest sink nodes
    connections_added = 0
    
    for terminal in true_terminals:
        if terminal not in pos:
            continue
            
        terminal_pos = pos[terminal]
        terminal_point = Point(terminal_pos)
        
        # Calculate distances from this terminal to all sinks
        sinks_with_distances = []
        for sink in sink_nodes:
            if sink in pos:
                sink_pos = pos[sink]
                distance = Point(sink_pos).distance(terminal_point)
                
                # Only consider sinks within the maximum distance
                if distance <= max_distance:
                    sinks_with_distances.append((sink, distance))
        
        # Sort sinks by distance to this terminal
        sinks_with_distances.sort(key=lambda x: x[1])
        
        # Take only the closest few sinks for this terminal
        closest_sinks = sinks_with_distances[:max_connections_per_terminal]
        
        # Connect the terminal to these closest sinks
        for sink, distance in closest_sinks:
            # Don't connect if already connected
            if not G.has_edge(terminal, sink):
                G.add_edge(terminal, sink,
                          road_name="evacuation_connector",
                          road_type="E",  # E for evacuation connector
                          weight=distance)
                connections_added += 1
                print(f"Connected terminal {terminal} to sink {sink} (distance: {distance:.2f})")
    
    print(f"Added {connections_added} targeted connections from dead-end nodes to sink nodes")
    print(f"Average connections per terminal: {connections_added / len(true_terminals) if true_terminals else 0:.1f}")
    
    total_time = time.time() - start_time
    print(f"Dead-end to sink connections added in {total_time:.4f} seconds")
    return G

def connect_sink_nodes_to_intersections(G, max_distance=100000):
    """
    Connect each sink node to the closest intersection node.
    
    Parameters:
    -----------
    G : networkx.Graph
        The road network graph with sink nodes and intersection nodes
    max_distance : float
        Maximum distance to consider for connections (default: 100000 units)
    
    Returns:
    --------
    G : networkx.Graph
        Updated graph with sink nodes connected to intersection nodes
    """
    print(f"Connecting sink nodes to closest intersection nodes...")
    start_time = time.time()
    
    # Get all node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Get all sink nodes
    sink_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'sink']
    if not sink_nodes:
        print("No sink nodes found in the graph. Nothing to connect.")
        return G
    
    print(f"Found {len(sink_nodes)} sink nodes to connect to intersections")
    
    # Get all intersection nodes
    intersection_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'intersection']
    if not intersection_nodes:
        print("No intersection nodes found in the graph. Cannot connect sink nodes.")
        return G
    
    print(f"Found {len(intersection_nodes)} intersection nodes as potential connection points")
    
    # For each sink node, find and connect to the closest intersection
    connections_added = 0
    
    for sink in sink_nodes:
        if sink not in pos:
            continue
            
        sink_pos = pos[sink]
        sink_point = Point(sink_pos)
        
        # Calculate distances from this sink to all intersections
        intersections_with_distances = []
        for intersection in intersection_nodes:
            if intersection in pos:
                intersection_pos = pos[intersection]
                distance = Point(intersection_pos).distance(sink_point)
                
                # Only consider intersections within the maximum distance
                if distance <= max_distance:
                    intersections_with_distances.append((intersection, distance))
        
        if not intersections_with_distances:
            print(f"Warning: No intersections found within range for sink {sink}")
            continue
            
        # Sort intersections by distance to this sink
        intersections_with_distances.sort(key=lambda x: x[1])
        
        # Get the closest intersection
        closest_intersection, distance = intersections_with_distances[0]
        
        # Connect the sink to the closest intersection
        if not G.has_edge(sink, closest_intersection):
            G.add_edge(sink, closest_intersection,
                      road_name="sink_to_intersection",
                      road_type="E",  # E for evacuation connector
                      weight=distance)
            connections_added += 1
            print(f"Connected sink {sink} to intersection {closest_intersection} (distance: {distance:.2f})")
    
    print(f"Added {connections_added} connections from sink nodes to intersection nodes")
    
    total_time = time.time() - start_time
    print(f"Sink to intersection connections added in {total_time:.4f} seconds")
    return G

def add_sink_nodes_to_graph(G, sink_nodes_gdf, routes_gdf):
    """
    Add sink nodes to the graph and connect them to the nearest road endpoint (start or end)
    
    Parameters:
    -----------
    G : networkx.Graph
        The existing road network graph
    sink_nodes_gdf : GeoDataFrame
        GeoDataFrame with sink node points (evacuation destinations)
    routes_gdf : GeoDataFrame
        Original routes GeoDataFrame used for finding connections
    """
    print("Adding sink nodes to the graph...")
    start_time = time.time()
    
    sink_connections = 0
    
    # First add all sink nodes as nodes
    for idx, row in sink_nodes_gdf.iterrows():
        # Create a unique ID for each sink node
        sink_id = f"s{idx}"
        
        # Add the sink node as a node
        G.add_node(sink_id, 
                 pos=(row.geometry.x, row.geometry.y),
                 node_type='sink',
                 geometry=row.geometry,
                 zip_code=row.get('ZIP_CODE', ''),
                 name=row.get('NAME', f'Sink {idx}'))
    
    print(f"Added {len(sink_nodes_gdf)} sink nodes")
    
    # Connect each sink node to the nearest endpoint of the nearest road
    for idx, sink in sink_nodes_gdf.iterrows():
        sink_id = f"s{idx}"
        sink_point = sink.geometry
        
        # Find the nearest road segments
        distances = routes_gdf.geometry.apply(lambda g: g.distance(sink_point))
        nearest_route_idx = distances.idxmin()
        nearest_route = routes_gdf.loc[nearest_route_idx]
        
        # Get both start and end nodes from G
        start_node = f"r{nearest_route_idx}_start"
        end_node = f"r{nearest_route_idx}_end"
        
        # Calculate distance to both endpoints and connect to the closer one
        connections_added = 0
        
        start_dist = float('inf')
        end_dist = float('inf')
        
        # Check and calculate distance to start node
        if start_node in G.nodes:
            start_pos = G.nodes[start_node]['pos']
            start_dist = Point(start_pos).distance(sink_point)
        
        # Check and calculate distance to end node
        if end_node in G.nodes:
            end_pos = G.nodes[end_node]['pos']
            end_dist = Point(end_pos).distance(sink_point)
        
        # Connect to the closest endpoint
        if start_dist <= end_dist and start_node in G.nodes:
            # Start node is closer
            G.add_edge(sink_id, start_node,
                      road_name="sink_connector",
                      road_type="S",  # S for sink connector
                      weight=start_dist)
            connections_added += 1
            print(f"Sink {idx} connected to start node of route {nearest_route_idx}")
        elif end_dist < start_dist and end_node in G.nodes:
            # End node is closer
            G.add_edge(sink_id, end_node,
                      road_name="sink_connector",
                      road_type="S",  # S for sink connector
                      weight=end_dist)
            connections_added += 1
            print(f"Sink {idx} connected to end node of route {nearest_route_idx}")
        
        # If no connections were made, try with the second nearest road
        if connections_added == 0:
            print(f"Warning: Could not connect sink node {idx} (ZIP {sink.get('ZIP_CODE', '')}) to nearest road")
            
            # Try second-closest road as fallback
            if len(distances) > 1:
                second_nearest = distances.nsmallest(2).index[1]
                start_node_2 = f"r{second_nearest}_start"
                end_node_2 = f"r{second_nearest}_end"
                
                # Calculate distances to both endpoints of second nearest road
                start_dist_2 = float('inf')
                end_dist_2 = float('inf')
                
                if start_node_2 in G.nodes:
                    start_pos_2 = G.nodes[start_node_2]['pos']
                    start_dist_2 = Point(start_pos_2).distance(sink_point)
                
                if end_node_2 in G.nodes:
                    end_pos_2 = G.nodes[end_node_2]['pos']
                    end_dist_2 = Point(end_pos_2).distance(sink_point)
                
                # Connect to the closest endpoint of second nearest road
                if start_dist_2 <= end_dist_2 and start_node_2 in G.nodes:
                    G.add_edge(sink_id, start_node_2,
                              road_name="sink_connector",
                              road_type="S",
                              weight=start_dist_2)
                    connections_added += 1
                    print(f"Sink {idx} connected to start node of second nearest route {second_nearest}")
                elif end_dist_2 < start_dist_2 and end_node_2 in G.nodes:
                    G.add_edge(sink_id, end_node_2,
                              road_name="sink_connector",
                              road_type="S",
                              weight=end_dist_2)
                    connections_added += 1
                    print(f"Sink {idx} connected to end node of second nearest route {second_nearest}")
        
        sink_connections += connections_added
    
    print(f"Added {sink_connections} connections from sink nodes to roads")
    
    total_time = time.time() - start_time
    print(f"Sink nodes added in {total_time:.4f} seconds")
    return G

def find_perimeter_nodes(G, min_distance_between_nodes=5000):
    """
    Find true perimeter nodes with improved detection.
    
    Parameters:
    -----------
    G : networkx.Graph
        The road network graph with node positions
    min_distance_between_nodes : float
        Minimum distance required between selected perimeter nodes
        
    Returns:
    --------
    list
        List of node IDs that are on the outer perimeter
    """
    from scipy.spatial import ConvexHull, Delaunay
    import numpy as np
    from shapely.geometry import MultiPoint, Point, Polygon, LineString
    
    print("Identifying perimeter nodes with comprehensive detection...")
    
    # Get all node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Remove sink nodes from consideration
    non_sink_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') != 'sink' and n in pos]
    print(f"Considering {len(non_sink_nodes)} non-sink nodes for perimeter detection")
    
    # Get positions for these nodes
    points_array = np.array([pos[n] for n in non_sink_nodes])
    
    # STEP 1: Create a convex hull of ALL nodes to find the rough boundary
    hull = ConvexHull(points_array)
    hull_points = [tuple(points_array[i]) for i in hull.vertices]
    boundary_polygon = Polygon(hull_points)
    
    # STEP 2: Find nodes within buffer distance of the boundary
    buffer_distance = 3000  # Increased buffer to catch more potential perimeter nodes
    inner_polygon = boundary_polygon.buffer(-buffer_distance)
    perimeter_zone = boundary_polygon.difference(inner_polygon)
    
    # STEP 3: Collect multiple types of perimeter node candidates
    perimeter_candidates = []
    
    # Type 1: Degree=1 nodes in perimeter zone
    degree_one_nodes = [node for node in non_sink_nodes if G.degree[node] == 1]
    for node in degree_one_nodes:
        point = Point(pos[node])
        if perimeter_zone.contains(point):
            perimeter_candidates.append((node, point, 'terminal'))
    
    print(f"Found {len(perimeter_candidates)} terminal nodes in perimeter zone")
    
    # Type 2: Directional scan - find outermost nodes in multiple directions
    center = np.mean(points_array, axis=0)
    directions = 3600  # Check in more directions to catch more points
    
    for angle in np.linspace(0, 2*np.pi, directions, endpoint=False):
        # Vector pointing in the current direction
        direction = np.array([np.cos(angle), np.sin(angle)])
        
        # Project all points onto this direction vector
        projections = np.dot(points_array - center, direction)
        
        # Find the index of the farthest point in this direction
        max_idx = np.argmax(projections)
        node = non_sink_nodes[max_idx]
        point = Point(pos[node])
        
        # Add this node to candidates if not already there
        if not any(n == node for n, _, _ in perimeter_candidates):
            perimeter_candidates.append((node, point, 'directional'))
    
    # Type 3: Degree=2 nodes that form sharp corners in the network
    for node in non_sink_nodes:
        if G.degree[node] == 2 and perimeter_zone.contains(Point(pos[node])):
            # Get the two connected nodes
            neighbors = list(G.neighbors(node))
            if len(neighbors) == 2 and neighbors[0] in pos and neighbors[1] in pos:
                v1 = np.array(pos[neighbors[0]]) - np.array(pos[node])
                v2 = np.array(pos[neighbors[1]]) - np.array(pos[node])
                
                # Normalize vectors
                v1 = v1 / np.linalg.norm(v1)
                v2 = v2 / np.linalg.norm(v2)
                
                # Calculate angle between vectors (dot product)
                angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
                
                # If angle is sharp (less than ~120 degrees), consider it a corner
                if angle < 2.1:  # radians (~120 degrees)
                    perimeter_candidates.append((node, Point(pos[node]), 'corner'))
    
    print(f"Total perimeter candidates: {len(perimeter_candidates)}")
    
    # STEP 4: Filter to ensure minimum distance between selected nodes
    selected_perimeter_nodes = []
    
    # First add all terminal nodes (highest priority)
    terminal_candidates = [(n, p) for n, p, t in perimeter_candidates if t == 'terminal']
    while terminal_candidates:
        node, point = terminal_candidates.pop(0)
        selected_perimeter_nodes.append(node)
        
        # Filter out nearby candidates
        terminal_candidates = [
            (n, p) for n, p in terminal_candidates
            if p.distance(point) >= min_distance_between_nodes
        ]
    
    # Then process other types of candidates
    other_candidates = [(n, p) for n, p, t in perimeter_candidates 
                       if t != 'terminal' and n not in selected_perimeter_nodes]
    
    while other_candidates:
        node, point = other_candidates.pop(0)
        
        # Check if it's too close to any already selected node
        too_close = False
        for selected_node in selected_perimeter_nodes:
            if point.distance(Point(pos[selected_node])) < min_distance_between_nodes:
                too_close = True
                break
                
        if not too_close:
            selected_perimeter_nodes.append(node)
    
    print(f"Selected {len(selected_perimeter_nodes)} perimeter nodes after filtering")
    return selected_perimeter_nodes

def build_graph_from_scratch(routes_path, intersections_path, centroids_path, sink_nodes_path):
    """Build complete graph from source files"""
    print("Building new graph from scratch...")
    start_time = time.time()
    
    # Load data
    print("Loading data...")
    routes_gdf, intersections_gdf = load_data(routes_path, intersections_path)
    
    # Load zip code centroids
    print("Loading zip code centroids...")
    centroids_gdf = load_centroids(centroids_path)
    print(f"Loaded {len(centroids_gdf)} zip code centroids")
    
    # Load sink nodes
    print("Loading sink nodes...")
    sink_nodes_gdf = load_sink_nodes(sink_nodes_path)
    print(f"Loaded {len(sink_nodes_gdf)} sink nodes")
    
    # Create graph
    print("Creating graph topology...")
    G = build_topological_graph(routes_gdf, intersections_gdf)
    
    # Add centroids to the graph
    print("Adding zip code centroids to graph...")
    G = add_centroids_to_graph(G, centroids_gdf, routes_gdf)
    
    # Add sink nodes to the graph
    print("Adding sink nodes to graph...")
    G = add_sink_nodes_to_graph(G, sink_nodes_gdf, routes_gdf)
    
    # Connect terminal nodes to nearest sink nodes
    G = connect_terminal_nodes_to_sinks(G, routes_gdf)
    
    build_time = time.time() - start_time
    print(f"Graph built from scratch in {build_time:.4f} seconds")
    
    return G

def main():
    # Define data paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    routes_path = os.path.join(data_dir, 'Evacuation_Routes_Tampa.geojson')
    intersections_path = os.path.join(data_dir, 'Intersection.geojson')
    centroids_path = os.path.join(data_dir, 'zip_code_centroids.json')
    sink_nodes_path = os.path.join(data_dir, 'sink_nodes.json')
    output_path = os.path.join(os.path.dirname(data_dir), 'evacuation_network.png')
    perimeter_output_path = os.path.join(os.path.dirname(data_dir), 'evacuation_network_perimeter.png')
    pickle_path = os.path.join(data_dir, 'evacuation_graph.pickle')
    
    start_time = time.time()
    
    # Check if pickled graph exists and load it if available
    if os.path.exists(pickle_path):
        print(f"Loading previously built graph from {pickle_path}...")
        try:
            with open(pickle_path, 'rb') as f:
                G = pickle.load(f)
            print(f"Graph loaded successfully: {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            
            # Also need to load GeoDataFrames for visualization
            routes_gdf = gpd.read_file(routes_path).to_crs(epsg=3857)
            intersections_gdf = gpd.read_file(intersections_path).to_crs(epsg=3857)
            
            print("Using cached graph from pickle file")
        except Exception as e:
            print(f"Error loading pickled graph: {e}")
            print("Building new graph instead...")
            G = build_graph_from_scratch(routes_path, intersections_path, centroids_path, sink_nodes_path)
    else:
        print("No pickled graph found. Building from scratch...")
        G = build_graph_from_scratch(routes_path, intersections_path, centroids_path, sink_nodes_path)
        
        # Save the newly built graph
        print(f"Saving graph to pickle file: {pickle_path}")
        with open(pickle_path, 'wb') as f:
            pickle.dump(G, f)
        print("Graph saved successfully")
    
    # Find perimeter nodes
    perimeter_nodes = find_perimeter_nodes(G)
    
    # Visualization with perimeter nodes
    print("Visualizing graph with perimeter nodes...")
    visualize_with_perimeter(G, perimeter_nodes, routes_gdf, intersections_gdf, perimeter_output_path)
    
    # Standard visualization
    print("Visualizing graph...")
    visualize_graph(G, routes_gdf, intersections_gdf, output_path)
    
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.4f} seconds")
    print("Done!")
    return G, perimeter_nodes

if __name__ == "__main__":
    G = main()