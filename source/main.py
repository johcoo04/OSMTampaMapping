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
    
    # Draw sink nodes as stars
    sink_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'sink']
    if sink_nodes:
        sink_scatter = ax.scatter(
            [pos[n][0] for n in sink_nodes if n in pos],
            [pos[n][1] for n in sink_nodes if n in pos],
            s=200, c='yellow', alpha=1.0, marker='*', edgecolor='black', label='Evacuation Points'
        )
    
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
    
    total_time = time.time() - start_time
    print(f"Visualization completed in {total_time:.4f} seconds")

def identify_all_road_tips(G):
    """
    Identify all road tip nodes without any filtering.
    
    Parameters:
    -----------
    G : networkx.Graph
        The road network graph
        
    Returns:
    --------
    list
        List of node IDs that are at the tips of roads
    """
    print("Identifying all road tips without filtering...")
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Find all nodes in the road network
    road_network = G.copy()
    
    # Remove existing connections to sinks and centroids
    edges_to_remove = []
    for u, v, data in road_network.edges(data=True):
        road_type = data.get('road_type', '')
        if road_type in ['Z', 'S', 'E']:  # Centroid, Sink, or Evacuation connectors
            edges_to_remove.append((u, v))
    
    road_network.remove_edges_from(edges_to_remove)
    
    # Find all terminal nodes (degree=1) in the road network
    all_tips = []
    
    for node, degree in road_network.degree():
        # Look for nodes with degree=1 (only one connection) in the road network
        if degree == 1:
            node_type = G.nodes[node].get('node_type', '')
            
            # Make sure it's a route endpoint
            if node_type == 'route_endpoint' and node in pos:
                all_tips.append(node)
    
    print(f"Found {len(all_tips)} road tips in the network")
    return all_tips

def visualize_with_road_tips(G, road_tips, routes_gdf, intersections_gdf, output_path=None):
    """Visualize the graph with road tips highlighted"""
    start_time = time.time()
    print("Starting visualization with road tips highlighted...")
    
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
            if road_type in ['I', 'J', 'E']:  # Skip evacuation connectors
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
    
    # Draw sink nodes as stars
    sink_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'sink']
    if sink_nodes:
        sink_scatter = ax.scatter(
            [pos[n][0] for n in sink_nodes if n in pos],
            [pos[n][1] for n in sink_nodes if n in pos],
            s=200, c='yellow', alpha=1.0, marker='*', edgecolor='black', label='Evacuation Points'
        )
    
    # Highlight road tips
    tips_positions = [pos[n] for n in road_tips if n in pos]
    if tips_positions:
        x_coords = [p[0] for p in tips_positions]
        y_coords = [p[1] for p in tips_positions]
        tips_scatter = ax.scatter(
            x_coords, y_coords, 
            s=150, c='green', marker='o', edgecolor='black', linewidth=1, label='Road Tips'
        )
    
    # Add legend with road types and node types
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
    
    # Set title and labels
    ax.set_title('Tampa Evacuation Network with Road Tips Highlighted')
    
    # Remove axes
    ax.set_axis_off()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to {output_path}")
    
    total_time = time.time() - start_time
    print(f"Visualization with road tips completed in {total_time:.4f} seconds")

def visualize_with_labeled_road_tips(G, road_tips, routes_gdf, intersections_gdf, output_path=None):
    """Visualize the graph with all road tips labeled with their IDs"""
    start_time = time.time()
    print("Starting visualization with labeled road tips...")
    
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
            if road_type in ['I', 'J', 'E']:  # Skip evacuation connectors
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
    
    # Draw road tips with labels
    tips_positions = [pos[n] for n in road_tips if n in pos]
    if tips_positions:
        x_coords = [p[0] for p in tips_positions]
        y_coords = [p[1] for p in tips_positions]
        tips_scatter = ax.scatter(
            x_coords, y_coords, 
            s=150, c='green', marker='o', edgecolor='black', linewidth=1, label='Road Tips'
        )
        
        # Add labels to road tips
        for i, tip in enumerate(road_tips):
            if tip in pos:
                # Use shortened labels to avoid clutter
                # Extract numeric part to make labels shorter
                if '_' in tip:
                    label = tip.split('_')[0].replace('r', '')
                else:
                    label = str(i)
                    
                ax.annotate(
                    label, 
                    xy=pos[tip],
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=10,
                    color='black',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7)
                )
    
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
    
    # Add legend with road types and node types
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
    
    # Set title and labels
    ax.set_title('Tampa Evacuation Network with Labeled Road Tips')
    
    # Remove axes
    ax.set_axis_off()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to {output_path}")
    
    # Also create a mapping from labels to road tip IDs and save it
    label_to_id = {}
    for i, tip in enumerate(road_tips):
        if '_' in tip:
            label = tip.split('_')[0].replace('r', '')
        else:
            label = str(i)
        label_to_id[label] = tip
    
    label_mapping_path = output_path.replace('.png', '_labels.json')
    with open(label_mapping_path, 'w') as f:
        json.dump(label_to_id, f, indent=2)
    print(f"Label mapping saved to {label_mapping_path}")
    
    total_time = time.time() - start_time
    print(f"Visualization with labeled road tips completed in {total_time:.4f} seconds")

def save_road_tips_to_file(G, road_tips, output_path):
    """
    Save all road tip nodes to a JSON file for later use.
    
    Parameters:
    -----------
    G : networkx.Graph
        The road network graph
    road_tips : list
        List of node IDs for all road tips
    output_path : str
        Path to save the JSON file
    """
    print(f"Saving {len(road_tips)} road tips to {output_path}...")
    
    # Get positions for road tips
    pos = nx.get_node_attributes(G, 'pos')
    
    # Create a dictionary with road tip data
    road_tips_data = {}
    
    for tip in road_tips:
        if tip in pos:
            road_tips_data[tip] = {
                "x": pos[tip][0],
                "y": pos[tip][1]
            }
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(road_tips_data, f, indent=2)
    
    print(f"Successfully saved {len(road_tips_data)} road tips to {output_path}")

def save_selected_road_tips(G, selected_tip_ids, output_path):
    """
    Save specific selected road tips to a JSON file.
    
    Parameters:
    -----------
    G : networkx.Graph
        The road network graph
    selected_tip_ids : list
        List of node IDs for selected road tips
    output_path : str
        Path to save the JSON file
    """
    print(f"Saving {len(selected_tip_ids)} selected road tips to {output_path}...")
    
    # Get positions for road tips
    pos = nx.get_node_attributes(G, 'pos')
    
    # Create a dictionary with road tip data
    selected_tips_data = {}
    
    for tip in selected_tip_ids:
        if tip in pos:
            selected_tips_data[tip] = {
                "x": pos[tip][0],
                "y": pos[tip][1]
            }
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(selected_tips_data, f, indent=2)
    
    print(f"Successfully saved {len(selected_tips_data)} selected road tips to {output_path}")

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
    
    build_time = time.time() - start_time
    print(f"Graph built from scratch in {build_time:.4f} seconds")
    
    return G, routes_gdf, intersections_gdf

def visualize_with_selected_road_tips(G, selected_tips, routes_gdf, intersections_gdf, output_path=None):
    """Visualize the graph with only the selected road tips highlighted"""
    start_time = time.time()
    print("Starting visualization with selected terminal nodes...")
    
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
            if road_type in ['I', 'J', 'E']:  # Skip evacuation connectors
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
    
    # Draw sink nodes as stars
    sink_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'sink']
    if sink_nodes:
        sink_scatter = ax.scatter(
            [pos[n][0] for n in sink_nodes if n in pos],
            [pos[n][1] for n in sink_nodes if n in pos],
            s=200, c='yellow', alpha=1.0, marker='*', edgecolor='black', label='Evacuation Points'
        )
    
    # Highlight selected road tips
    selected_positions = [pos[n] for n in selected_tips if n in pos]
    if selected_positions:
        x_coords = [p[0] for p in selected_positions]
        y_coords = [p[1] for p in selected_positions]
        tips_scatter = ax.scatter(
            x_coords, y_coords, 
            s=200, c='red', marker='D', edgecolor='black', linewidth=2, label='Selected Terminal Nodes'
        )
    
    # Add legend with road types and node types
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
    
    # Set title and labels
    ax.set_title('Tampa Evacuation Network with Selected Terminal Nodes')
    
    # Remove axes
    ax.set_axis_off()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to {output_path}")
    
    total_time = time.time() - start_time
    print(f"Visualization with selected terminal nodes completed in {total_time:.4f} seconds")

def main():
    # Define data paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    routes_path = os.path.join(data_dir, 'Evacuation_Routes_Tampa.geojson')
    intersections_path = os.path.join(data_dir, 'Intersection.geojson')
    centroids_path = os.path.join(data_dir, 'zip_code_centroids.json')
    sink_nodes_path = os.path.join(data_dir, 'sink_nodes.json')
    output_path = os.path.join(os.path.dirname(data_dir), 'evacuation_network.png')
    road_tips_output_path = os.path.join(os.path.dirname(data_dir), 'evacuation_network_road_tips.png')
    labeled_tips_output_path = os.path.join(os.path.dirname(data_dir), 'evacuation_network_labeled_tips.png')
    pickle_path = os.path.join(data_dir, 'evacuation_graph.pickle')
    road_tips_path = os.path.join(data_dir, 'road_tips.json')
    selected_tips_path = os.path.join(data_dir, 'selected_road_tips.json')
    selected_tips_output_path = os.path.join(os.path.dirname(data_dir), 'evacuation_network_with_sinks.png')
    
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
            
            # Load existing road tips if available
            if os.path.exists(road_tips_path):
                print(f"Loading existing road tips from {road_tips_path}...")
                with open(road_tips_path, 'r') as f:
                    road_tips_data = json.load(f)
                    road_tips = list(road_tips_data.keys())
                print(f"Loaded {len(road_tips)} road tips successfully")
            else:
                # Identify all road tips if not already saved
                road_tips = identify_all_road_tips(G)
                save_road_tips_to_file(G, road_tips, road_tips_path)
            
        except Exception as e:
            print(f"Error loading pickled graph: {e}")
            print("Building new graph instead...")
            G, routes_gdf, intersections_gdf = build_graph_from_scratch(routes_path, intersections_path, centroids_path, sink_nodes_path)
            
            # Identify all road tips
            road_tips = identify_all_road_tips(G)
            
            # Save road tips to a file
            save_road_tips_to_file(G, road_tips, road_tips_path)
            
            # Create visualization with labeled road tips
            print("Visualizing graph with labeled road tips...")
            visualize_with_labeled_road_tips(G, road_tips, routes_gdf, intersections_gdf, labeled_tips_output_path)
            
            # Visualization with road tips
            print("Visualizing graph with road tips...")
            visualize_with_road_tips(G, road_tips, routes_gdf, intersections_gdf, road_tips_output_path)
    else:
        print("No pickled graph found. Building from scratch...")
        G, routes_gdf, intersections_gdf = build_graph_from_scratch(routes_path, intersections_path, centroids_path, sink_nodes_path)
        
        # Save the newly built graph
        print(f"Saving graph to pickle file: {pickle_path}")
        with open(pickle_path, 'wb') as f:
            pickle.dump(G, f)
        print("Graph saved successfully")
        
        # Identify all road tips
        road_tips = identify_all_road_tips(G)
        
        # Save road tips to a file
        save_road_tips_to_file(G, road_tips, road_tips_path)
        
        # Create visualization with labeled road tips
        print("Visualizing graph with labeled road tips...")
        visualize_with_labeled_road_tips(G, road_tips, routes_gdf, intersections_gdf, labeled_tips_output_path)
        
        # Visualization with road tips
        print("Visualizing graph with road tips...")
        visualize_with_road_tips(G, road_tips, routes_gdf, intersections_gdf, road_tips_output_path)
    
    # Standard visualization
    print("Visualizing graph...")
    visualize_graph(G, routes_gdf, intersections_gdf, output_path)
    
    # After viewing the labeled visualization
    # Load the label mapping
    label_mapping_path = labeled_tips_output_path.replace('.png', '_labels.json')
    with open(label_mapping_path, 'r') as f:
        label_to_id = json.load(f)

    # Select the road tips you want by their labels
    selected_labels = ["2332", "2322", "298", "606", "2082", "1653", "1224", "3383", 
                       "2291", "2355", "1654", "1780", "2261", "646", "2260", "1789", 
                       "2430", "192", "54", "1802", "473", "1412", "122", "1064", "989"]
    
    # Convert labels to strings if they're not already
    selected_labels = [str(label) for label in selected_labels]
    
    # Get the corresponding IDs
    selected_tip_ids = [label_to_id[label] for label in selected_labels if label in label_to_id]
    
    # Save the selected road tips
    save_selected_road_tips(G, selected_tip_ids, selected_tips_path)
    print(f"Saved {len(selected_tip_ids)} selected road tips to {selected_tips_path}")
    
    # Visualize the selected road tips (terminal nodes)
    print("Visualizing graph with selected terminal nodes...")
    visualize_with_selected_road_tips(G, selected_tip_ids, routes_gdf, intersections_gdf, selected_tips_output_path)
    
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.4f} seconds")
    print("Done!")
    return G, road_tips

if __name__ == "__main__":
    G, road_tips = main()