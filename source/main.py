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
    Add centroids to the graph and connect them to the nearest roads
    
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
    
    # Connect each centroid to the nearest roads
    for idx, centroid in centroids_gdf.iterrows():
        centroid_id = f"c{idx}"
        centroid_point = centroid.geometry
        
        # Find the nearest road segments (always get at least one)
        distances = routes_gdf.geometry.apply(lambda g: g.distance(centroid_point))
        nearest_route_idx = distances.idxmin()  # Get the closest road
        nearest_route = routes_gdf.loc[nearest_route_idx]
        
        # Get ONLY the start node from G
        start_node = f"r{nearest_route_idx}_start"
        
        # Connect to start node only
        connections_added = 0
        if start_node in G.nodes:  # Ensure the node exists
            node_pos = G.nodes[start_node]['pos']
            dist = Point(node_pos).distance(centroid_point)
            
            # Add edge with attributes
            G.add_edge(centroid_id, start_node,
                      road_name="centroid_connector",
                      road_type="Z",  # Z for centroid connector
                      weight=dist)
            connections_added += 1
        
        if connections_added == 0:
            print(f"Warning: Could not connect centroid {idx} (ZIP {centroid.get('ZIP_CODE', '')}) to any road")
            
            # Try second-closest road as fallback
            if len(distances) > 1:
                second_nearest = distances.nsmallest(2).index[1]
                start_node_2 = f"r{second_nearest}_start"
                
                if start_node_2 in G.nodes:
                    node_pos = G.nodes[start_node_2]['pos']
                    dist = Point(node_pos).distance(centroid_point)
                    G.add_edge(centroid_id, start_node_2,
                              road_name="centroid_connector",
                              road_type="Z",
                              weight=dist)
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
        
        # Add zip code labels next to each centroid
        for node in centroid_nodes:
            if node in pos:
                x, y = pos[node]
                zip_code = G.nodes[node].get('zip_code', '')
                if zip_code:
                    # Position label slightly offset from the dot
                    ax.text(x + 1000, y + 1000, zip_code, 
                           fontsize=12, fontweight='bold', 
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Draw sink nodes as stars
    sink_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'sink']
    if sink_nodes:
        sink_scatter = ax.scatter(
            [pos[n][0] for n in sink_nodes if n in pos],
            [pos[n][1] for n in sink_nodes if n in pos],
            s=200, c='yellow', alpha=1.0, marker='*', edgecolor='black', label='Evacuation Points'
        )
        
        # Add zip code labels next to each sink node
        for node in sink_nodes:
            if node in pos:
                x, y = pos[node]
                zip_code = G.nodes[node].get('zip_code', '')
                if zip_code:
                    # Position label slightly offset from the star
                    ax.text(x + 1000, y + 1000, zip_code, 
                           fontsize=14, fontweight='bold', color='red',
                           bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
    
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
    Add sink nodes to the graph and connect them to the nearest roads
    
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
    
    # Connect each sink node to the nearest roads
    for idx, sink in sink_nodes_gdf.iterrows():
        sink_id = f"s{idx}"
        sink_point = sink.geometry
        
        # Find the nearest road segments
        distances = routes_gdf.geometry.apply(lambda g: g.distance(sink_point))
        nearest_route_idx = distances.idxmin()
        nearest_route = routes_gdf.loc[nearest_route_idx]
        
        # Get the start node from G
        start_node = f"r{nearest_route_idx}_start"
        
        # Connect to start node
        connections_added = 0
        if start_node in G.nodes:
            node_pos = G.nodes[start_node]['pos']
            dist = Point(node_pos).distance(sink_point)
            
            # Add edge with attributes
            G.add_edge(sink_id, start_node,
                      road_name="sink_connector",
                      road_type="S",  # S for sink connector
                      weight=dist)
            connections_added += 1
        
        if connections_added == 0:
            print(f"Warning: Could not connect sink node {idx} (ZIP {sink.get('ZIP_CODE', '')}) to any road")
            
            # Try second-closest road as fallback
            if len(distances) > 1:
                second_nearest = distances.nsmallest(2).index[1]
                start_node_2 = f"r{second_nearest}_start"
                
                if start_node_2 in G.nodes:
                    node_pos = G.nodes[start_node_2]['pos']
                    dist = Point(node_pos).distance(sink_point)
                    G.add_edge(sink_id, start_node_2,
                              road_name="sink_connector",
                              road_type="S",
                              weight=dist)
                    connections_added += 1
        
        sink_connections += connections_added
    
    print(f"Added {sink_connections} connections from sink nodes to roads")
    
    total_time = time.time() - start_time
    print(f"Sink nodes added in {total_time:.4f} seconds")
    return G

def main():
    # Define data paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    routes_path = os.path.join(data_dir, 'Evacuation_Routes_Tampa.geojson')
    intersections_path = os.path.join(data_dir, 'Intersection.geojson')
    centroids_path = os.path.join(data_dir, 'zip_code_centroids.json')
    sink_nodes_path = os.path.join(data_dir, 'sink_nodes.json')  # Path to sink nodes file
    output_path = os.path.join(os.path.dirname(data_dir), 'evacuation_network_with_sinks.png')
    pickle_path = os.path.join(data_dir, 'evacuation_graph_with_sinks.pickle')
    
    start_time = time.time()
    
    print("Loading data...")
    routes_gdf, intersections_gdf = load_data(routes_path, intersections_path)
    loading_time = time.time() - start_time
    print(f"Data loading completed in {loading_time:.4f} seconds")
    
    # Load zip code centroids
    print("Loading zip code centroids...")
    centroids_gdf = load_centroids(centroids_path)
    print(f"Loaded {len(centroids_gdf)} zip code centroids")
    
    # Load sink nodes
    print("Loading sink nodes...")
    sink_nodes_gdf = load_sink_nodes(sink_nodes_path)
    print(f"Loaded {len(sink_nodes_gdf)} sink nodes")
    
    print("Creating graph...")
    G = build_topological_graph(routes_gdf, intersections_gdf)
    
    # Add centroids to the graph
    print("Adding zip code centroids to graph...")
    G = add_centroids_to_graph(G, centroids_gdf, routes_gdf)
    
    # Add sink nodes to the graph
    print("Adding sink nodes to graph...")
    G = add_sink_nodes_to_graph(G, sink_nodes_gdf, routes_gdf)
    
    # Save graph to pickle file
    print(f"Saving graph to pickle file: {pickle_path}")
    with open(pickle_path, 'wb') as f:
        pickle.dump(G, f)
    print("Graph saved successfully")
    
    print("Visualizing graph...")
    visualize_graph(G, routes_gdf, intersections_gdf, output_path)
    
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.4f} seconds")
    print("Done!")
    return G

if __name__ == "__main__":
    G = main()