import os
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from rtree import index  # You might need to pip install rtree

def load_data(routes_path, centroids_path, sink_nodes_path):
    """Load evacuation routes, zip code centroids, and sink nodes."""
    print("Loading data...")
    
    # Load evacuation routes
    routes_gdf = gpd.read_file(routes_path).to_crs(epsg=3857)
    print(f"Loaded {len(routes_gdf)} evacuation routes")
    
    # Load zip code centroids
    with open(centroids_path, 'r') as f:
        centroids_data = json.load(f)
    centroids = [
        {"geometry": Point(coords["centroid_x"], coords["centroid_y"]), "ZIP_CODE": zip_code}
        for zip_code, coords in centroids_data.items()
    ]
    centroids_gdf = gpd.GeoDataFrame(centroids, crs="EPSG:3857")
    print(f"Loaded {len(centroids_gdf)} zip code centroids")
    
    # Load sink nodes
    with open(sink_nodes_path, 'r') as f:
        sink_data = json.load(f)
    sinks = [
        {"geometry": Point(coords["centroid_x"], coords["centroid_y"]), "ZIP_CODE": zip_code}
        for zip_code, coords in sink_data.items()
    ]
    sinks_gdf = gpd.GeoDataFrame(sinks, crs="EPSG:3857")
    print(f"Loaded {len(sinks_gdf)} sink nodes")
    
    return routes_gdf, centroids_gdf, sinks_gdf

def create_connected_graph(routes_gdf, centroids_gdf, sinks_gdf, connection_threshold=100):
    """Create a connected graph with centroids, routes, and sinks."""
    print("Creating connected graph...")
    start_time = time.time()
    
    # Initialize graph
    G = nx.Graph()
    
    # First pass: Add all route points and connect consecutive points in each route
    print("Adding route segments to graph...")
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
    
    print(f"Added {len(routes_gdf)} routes with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    # Second pass: Build spatial index for all route nodes
    print("Building spatial index for route nodes...")
    route_nodes = [(node, data["pos"]) for node, data in G.nodes(data=True) if data["node_type"] == "route"]
    
    # Create spatial index
    spatial_idx = index.Index()
    for i, (node, pos) in enumerate(route_nodes):
        # Index format: (id, (left, bottom, right, top))
        spatial_idx.insert(i, (pos[0], pos[1], pos[0], pos[1]))
    
    # Third pass: Connect nearby route nodes
    print("Connecting nearby route nodes...")
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
    
    print(f"Made {connections_made} new connections between nearby route nodes")
    
    # Fourth pass: Check connectivity of route network
    print("Checking route network connectivity...")
    route_subgraph = G.subgraph([n for n, d in G.nodes(data=True) if d["node_type"] == "route"])
    
    if nx.is_connected(route_subgraph):
        print("Route network is fully connected!")
    else:
        components = list(nx.connected_components(route_subgraph))
        print(f"Warning: Route network has {len(components)} disconnected components")
        largest = max(components, key=len)
        print(f"Largest component has {len(largest)} nodes ({len(largest)/len(route_subgraph.nodes())*100:.1f}% of route nodes)")
        
        # If connectivity is very poor, we could try increasing the threshold and reconnecting
        if len(largest)/len(route_subgraph.nodes()) < 0.9 and connection_threshold < 500:
            print(f"Poor connectivity detected. Attempting to connect major components with increased threshold...")
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
                        print(f"Connected components by adding edge with distance {min_dist:.2f} meters")
                        comp1.extend(comp2)  # Merge components
    
    # Fifth pass: Add centroids and connect to multiple nearest route nodes
    print("Adding and connecting centroids...")
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
            print(f"Connected centroid {centroid['ZIP_CODE']} to {connection_count} route nodes")
        else:
            print(f"Warning: Could not find nearby route nodes for centroid {centroid['ZIP_CODE']}")
    
    # Sixth pass: Add sink nodes and connect to multiple nearest route nodes
    print("Adding and connecting sink nodes...")
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
            print(f"Connected sink {sink['ZIP_CODE']} to {connection_count} route nodes")
        else:
            print(f"Warning: Could not find nearby route nodes for sink {sink['ZIP_CODE']}")
    
    # Final check: Is the entire graph connected?
    print("Checking overall graph connectivity...")
    if nx.is_connected(G):
        print("Success! Graph is fully connected.")
    else:
        components = list(nx.connected_components(G))
        print(f"Warning: Graph has {len(components)} disconnected components")
        largest = max(components, key=len)
        print(f"Largest component has {len(largest)} nodes ({len(largest)/len(G.nodes())*100:.1f}% of all nodes)")
        
        # Report how many centroid and sink nodes are in the main component
        centroids_in_main = sum(1 for node in largest if G.nodes[node].get("node_type") == "centroid")
        sinks_in_main = sum(1 for node in largest if G.nodes[node].get("node_type") == "sink")
        print(f"Main component contains {centroids_in_main}/{len(centroids_gdf)} centroids and {sinks_in_main}/{len(sinks_gdf)} sinks")
        
        # Try to find and connect the remaining components
        if len(components) > 1:
            print("Attempting to connect remaining components...")
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
                    print(f"Connected component {comp_idx} to main component with edge of distance {min_dist:.2f} meters")
                    main_comp = main_comp.union(other_comp)  # Update main component
    
    # Final connectivity check after attempted fixes
    if not nx.is_connected(G):
        print("Warning: Graph remains disconnected after attempted fixes")
        
        # Identify which centroids and sinks are disconnected
        components = list(nx.connected_components(G))
        main_comp = max(components, key=len)
        
        disconnected_centroids = [G.nodes[node]["zip_code"] for node in G.nodes() 
                                if G.nodes[node].get("node_type") == "centroid" and node not in main_comp]
        disconnected_sinks = [G.nodes[node]["zip_code"] for node in G.nodes() 
                            if G.nodes[node].get("node_type") == "sink" and node not in main_comp]
        
        if disconnected_centroids:
            print(f"Disconnected centroids: {', '.join(disconnected_centroids)}")
        if disconnected_sinks:
            print(f"Disconnected sinks: {', '.join(disconnected_sinks)}")
    
    total_time = time.time() - start_time
    print(f"Graph creation completed in {total_time:.2f} seconds")
    return G

def visualize_path_with_road_types(G, path, output_path, centroid_zip=""):
    """Visualize the shortest path with color-coded road types."""
    print(f"Visualizing shortest path for {centroid_zip}...")
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
        alpha = 0.15 if road_type in ['CONNECTOR', 'CENTROID_CONNECTOR', 'SINK_CONNECTOR'] else 0.5
        width = 0.7 if road_type in ['CONNECTOR', 'CENTROID_CONNECTOR', 'SINK_CONNECTOR'] else 1.5
        
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
                if road_type in ['CONNECTOR', 'CENTROID_CONNECTOR', 'SINK_CONNECTOR', 'COMPONENT_CONNECTOR']:
                    color = 'yellow'
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                                      edge_color=color, width=3, alpha=1.0)
    
    # Add a legend
    legend_items = []
    
    # Road types for legend
    for road_type, color in road_type_colors.items():
        # Only show in legend if there are edges of this type
        if any(d.get('road_type') == road_type for _, _, d in G.edges(data=True)):
            if road_type in ['CONNECTOR', 'CENTROID_CONNECTOR', 'SINK_CONNECTOR', 'COMPONENT_CONNECTOR']:
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
    print(f"Visualization saved to {output_path}")
    plt.close()

def visualize_components(G, output_path):
    """Visualize the connected components of the graph."""
    print("Visualizing connected components...")
    pos = nx.get_node_attributes(G, 'pos')
    
    # Get connected components
    components = list(nx.connected_components(G))
    
    # Plot the graph
    plt.figure(figsize=(15, 15))
    
    # Use different colors for each component (up to 10 components)
    colors = plt.cm.tab10.colors
    
    # Draw each component with a different color
    for i, component in enumerate(sorted(components, key=len, reverse=True)):
        color = colors[i % len(colors)]
        subgraph = G.subgraph(component)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.6, edge_color=color)
        
        # FIX: Draw nodes with proper list to avoid warning
        nodes = list(subgraph.nodes())
        if nodes:  # Only draw if there are nodes
            nx.draw_networkx_nodes(subgraph, pos, nodelist=nodes, node_size=10, 
                                 node_color=[color] * len(nodes), alpha=0.8)
    
    # Draw centroids and sinks with special markers - FIX to avoid warning
    centroid_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "centroid"]
    if centroid_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=centroid_nodes, node_size=100, 
                             node_color=["green"] * len(centroid_nodes), 
                             node_shape='*', alpha=1.0)
    
    sink_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "sink"]
    if sink_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=sink_nodes, node_size=100, 
                             node_color=["red"] * len(sink_nodes), 
                             node_shape='s', alpha=1.0)
    
    # Add a legend
    legend_items = []
    legend_items.append(plt.Line2D([], [], marker='*', color='w', markerfacecolor='green', 
                       markersize=15, label='Centroids', linestyle='None'))
    legend_items.append(plt.Line2D([], [], marker='s', color='w', markerfacecolor='red', 
                       markersize=10, label='Sinks', linestyle='None'))
    
    # Add component legend items
    for i in range(min(len(components), 5)):
        comp_size = len(sorted(components, key=len, reverse=True)[i])
        legend_items.append(plt.Line2D([], [], marker='o', color='w', 
                           markerfacecolor=colors[i], markersize=8, 
                           label=f'Component {i+1} ({comp_size} nodes)', 
                           linestyle='None'))
    
    if len(components) > 5:
        other_nodes = sum(len(c) for c in sorted(components, key=len, reverse=True)[5:])
        legend_items.append(plt.Line2D([], [], marker='o', color='w', 
                           markerfacecolor='gray', markersize=8, 
                           label=f'Other components ({other_nodes} nodes)', 
                           linestyle='None'))
    
    plt.legend(handles=legend_items, loc='best')
    
    # Save the visualization
    plt.title(f"Graph Components ({len(components)} total)")
    plt.savefig(output_path, dpi=300)
    print(f"Component visualization saved to {output_path}")
    plt.close()

def process_all_centroids(G, centroids_gdf, output_dir):
    """Process all centroids, find the top 3 closest sinks for each."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sink nodes
    sink_nodes = [node for node, data in G.nodes(data=True) if data.get("node_type") == "sink"]
    
    # Check if the graph is connected
    if not nx.is_connected(G):
        print("Warning: Graph is not fully connected. Some paths may not be found.")
        # Get the components
        components = list(nx.connected_components(G))
        # Map each node to its component
        node_to_component = {}
        for i, component in enumerate(components):
            for node in component:
                node_to_component[node] = i
    
    # Process each centroid
    results = []
    for _, centroid in centroids_gdf.iterrows():
        zip_code = centroid["ZIP_CODE"]
        print(f"Processing centroid {zip_code}...")
        
        # Find the centroid node
        centroid_node = None
        for node, data in G.nodes(data=True):
            if data.get("node_type") == "centroid" and data.get("zip_code") == zip_code:
                centroid_node = node
                break
        
        if not centroid_node:
            print(f"Warning: Centroid {zip_code} not found in the graph!")
            continue
        
        # Find paths to all reachable sinks
        all_paths = []
        
        # If the graph is connected, search all sinks
        if nx.is_connected(G):
            for sink_node in sink_nodes:
                try:
                    path = nx.shortest_path(G, source=centroid_node, target=sink_node, weight="weight")
                    distance = nx.shortest_path_length(G, source=centroid_node, target=sink_node, weight="weight")
                    sink_zip = G.nodes[sink_node].get("zip_code")
                    
                    # Count road types in the path
                    path_edges = list(zip(path, path[1:]))
                    road_types_in_path = {}
                    for u, v in path_edges:
                        if G.has_edge(u, v):
                            road_type = G.edges[u, v].get('road_type', 'UNKNOWN')
                            road_types_in_path[road_type] = road_types_in_path.get(road_type, 0) + 1
                    
                    all_paths.append({
                        "sink": sink_zip,
                        "distance_meters": distance,
                        "distance_km": distance/1000,
                        "path_nodes": len(path),
                        "path": path,
                        "road_types": road_types_in_path
                    })
                except nx.NetworkXNoPath:
                    continue
        else:
            # If not connected, only search sinks in the same component
            centroid_component = node_to_component.get(centroid_node)
            component_sinks = [sink for sink in sink_nodes if node_to_component.get(sink) == centroid_component]
            
            for sink_node in component_sinks:
                try:
                    path = nx.shortest_path(G, source=centroid_node, target=sink_node, weight="weight")
                    distance = nx.shortest_path_length(G, source=centroid_node, target=sink_node, weight="weight")
                    sink_zip = G.nodes[sink_node].get("zip_code")
                    
                    # Count road types in the path
                    path_edges = list(zip(path, path[1:]))
                    road_types_in_path = {}
                    for u, v in path_edges:
                        if G.has_edge(u, v):
                            road_type = G.edges[u, v].get('road_type', 'UNKNOWN')
                            road_types_in_path[road_type] = road_types_in_path.get(road_type, 0) + 1
                    
                    all_paths.append({
                        "sink": sink_zip,
                        "distance_meters": distance,
                        "distance_km": distance/1000,
                        "path_nodes": len(path),
                        "path": path,
                        "road_types": road_types_in_path
                    })
                except nx.NetworkXNoPath:
                    continue
        
        # Sort paths by distance
        all_paths.sort(key=lambda x: x["distance_meters"])
        
        # Get top 3 paths (or fewer if not enough)
        top_paths = all_paths[:min(3, len(all_paths))]
        
        if top_paths:
            # Visualize only the shortest path
            shortest_path_data = top_paths[0]
            output_path = os.path.join(output_dir, f"path_{zip_code}_to_{shortest_path_data['sink']}.png")
            visualize_path_with_road_types(G, shortest_path_data["path"], output_path, zip_code)
            
            # Add result to collection
            result = {
                "centroid": zip_code,
                "paths": top_paths
            }
            results.append(result)
            
            print(f"Found {len(top_paths)} paths from centroid {zip_code}")
            for i, path_data in enumerate(top_paths):
                print(f"  Path {i+1}: to sink {path_data['sink']}, distance: {path_data['distance_km']:.2f} km")
        else:
            print(f"No path found from centroid {zip_code} to any sink!")
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
    
    print(f"Summary saved to {summary_path}")
    return results

def main():
    # Define file paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    routes_path = os.path.join(data_dir, "Evacuation_Routes_Tampa.geojson")
    centroids_path = os.path.join(data_dir, "zip_code_centroids.json")
    sink_nodes_path = os.path.join(data_dir, "sink_nodes.json")
    components_path = os.path.join(data_dir, "graph_components.png")
    results_dir = os.path.join(data_dir, "results")
    
    # Load data
    routes_gdf, centroids_gdf, sinks_gdf = load_data(routes_path, centroids_path, sink_nodes_path)
    
    # Create graph with improved connectivity
    # Adjust connection_threshold as needed - higher values create more connections
    G = create_connected_graph(routes_gdf, centroids_gdf, sinks_gdf, connection_threshold=150)
    
    # Visualize the components to diagnose connectivity issues
    visualize_components(G, components_path)
    
    # Process all centroids and find paths
    process_all_centroids(G, centroids_gdf, results_dir)

if __name__ == "__main__":
    main()