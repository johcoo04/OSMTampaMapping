import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import os
import pickle
from shapely.geometry import Point, LineString
import multiprocessing
from functools import partial
import numpy as np
import time

def load_data(routes_path, intersections_path):
    """Load GeoJSON data and standardize CRS"""
    routes_gdf = gpd.read_file(routes_path)
    intersections_gdf = gpd.read_file(intersections_path)
    
    # Ensure both have the same CRS (coordinate reference system)
    routes_gdf = routes_gdf.to_crs(epsg=3857)
    intersections_gdf = intersections_gdf.to_crs(epsg=3857)
    
    print(f"Loaded {len(routes_gdf)} evacuation routes")
    print(f"Loaded {len(intersections_gdf)} intersections")
    
    return routes_gdf, intersections_gdf

def process_route_chunk(args):
    """Process a chunk of routes and return edges with timing information"""
    chunk_df, intersections_gdf, process_id, snap_distance = args
    
    edges = []
    edge_count = 0
    start_time = time.time()
    last_time = start_time
    
    for idx, route in chunk_df.iterrows():
        if route.geometry.geom_type != 'LineString':
            continue
            
        # Find nearest intersections to start and end points of the route
        start_point = Point(route.geometry.coords[0])
        end_point = Point(route.geometry.coords[-1])
        
        # Calculate distances to all intersections
        start_distances = intersections_gdf.geometry.apply(lambda geom: geom.distance(start_point))
        end_distances = intersections_gdf.geometry.apply(lambda geom: geom.distance(end_point))
        
        # Find closest intersections within snap distance
        start_node = start_distances.idxmin() if start_distances.min() <= snap_distance else None
        end_node = end_distances.idxmin() if end_distances.min() <= snap_distance else None
        
        # Skip if we couldn't find both nodes
        if start_node is None or end_node is None or start_node == end_node:
            continue
            
        # Add edge with minimal attributes
        road_name = route.get('STREET', 'unnamed')  # Default to unnamed if STREET not available
        length = route.geometry.length
        
        edges.append((start_node, end_node, {
            'road_name': road_name,
            'weight': length,
            'geometry': route.geometry
        }))
        
        edge_count += 1
        
        # Print timing every 10 edges in real-time
        if edge_count % 10 == 0:
            current_time = time.time()
            delta = current_time - last_time
            print(f"Process {process_id}: Processed {edge_count} edges, last 10 took {delta:.4f} seconds")
            last_time = current_time
    
    # Print total timing for the chunk
    total_time = time.time() - start_time
    print(f"Process {process_id}: Total {edge_count} edges in {total_time:.4f} seconds")
    
    return edges

def create_graph(routes_gdf, intersections_gdf, snap_distance=50):
    """Create a networkx graph using intersections as nodes and routes as edges with multiprocessing"""
    G = nx.Graph()
    start_time = time.time()
    
    # Add intersections as nodes
    for idx, row in intersections_gdf.iterrows():
        G.add_node(idx, 
                   pos=(row.geometry.x, row.geometry.y), 
                   node_type='intersection',
                   geometry=row.geometry)
    
    # Process routes into edges using multiprocessing
    print("Processing routes into edges with multiprocessing...")
    
    # Determine number of processes to use 
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores for multiprocessing")
    
    # Split routes into chunks based on index ranges instead of using numpy array_split
    total_routes = len(routes_gdf)
    chunk_size = total_routes // num_cores
    chunks = []
    
    for i in range(num_cores):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_cores - 1 else total_routes
        chunk = routes_gdf.iloc[start_idx:end_idx].copy()
        chunks.append(chunk)
    
    # Prepare arguments for the process_route_chunk function
    chunk_args = [
        (chunk, intersections_gdf, i, snap_distance) 
        for i, chunk in enumerate(chunks)
    ]
    
    # Process chunks in parallel
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(process_route_chunk, chunk_args)
    
    # Flatten results and add edges to graph
    all_edges = []
    
    for edges in results:
        all_edges.extend(edges)
    
    G.add_edges_from(all_edges)
    
    total_time = time.time() - start_time
    print(f"\nTotal graph creation time: {total_time:.4f} seconds")
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Remove isolated nodes (nodes with no incoming or outgoing edges)
    isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0]
    G.remove_nodes_from(isolated_nodes)
    print(f"Removed {len(isolated_nodes)} isolated nodes")
    print(f"Final graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    return G

def build_topological_graph(routes_gdf, intersections_gdf):
    """Create a graph using true topological connections between roads"""
    print("Building graph with topological connections...")
    start_time = time.time()
    
    # Initialize new graph
    G = nx.Graph()
    
    # First add all intersections as nodes
    print("Adding intersection nodes...")
    for idx, row in intersections_gdf.iterrows():
        G.add_node(f"i{idx}", 
                   pos=(row.geometry.x, row.geometry.y), 
                   node_type='intersection',
                   geometry=row.geometry)
    
    # Create a unified lines dataset from all routes
    print("Processing routes for topological connections...")
    # Add endpoints of each route as nodes
    endpoint_nodes = {}  # Maps (x,y) coordinates to node IDs
    
    # Process each route and add its endpoints as nodes
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
    
    # Connect endpoints of different routes when they're very close to each other
    print("Connecting proximate route endpoints...")
    endpoint_connections = 0
    
    # Convert endpoint_nodes keys to Points for easier distance calculation
    endpoints = [(Point(coord), node_id) for coord, node_id in endpoint_nodes.items()]
    
    # Check each pair of endpoints
    for i, (point1, node1) in enumerate(endpoints):
        for point2, node2 in endpoints[i+1:]:
            # If endpoints are very close and not already connected
            if point1.distance(point2) < 0.001 and not G.has_edge(node1, node2):
                # Connect the endpoints
                G.add_edge(node1, node2,
                          road_name="junction", 
                          road_type="J", 
                          weight=point1.distance(point2))
                endpoint_connections += 1
    
    print(f"Added {endpoint_connections} connections between proximate route endpoints")
    
    # Find routes that cross each other and add connections
    print("Finding route intersections...")
    route_intersections = 0
    
    # Get all route edges
    route_edges = [(u, v, data) for u, v, data in G.edges(data=True) 
                   if 'geometry' in data and data.get('road_name') != 'connector']
    
    # Check each pair of routes
    for i, (u1, v1, data1) in enumerate(route_edges):
        geom1 = data1['geometry']
        
        for u2, v2, data2 in route_edges[i+1:]:
            geom2 = data2['geometry']
            
            # If routes intersect and aren't already connected
            if geom1.intersects(geom2) and not (
                G.has_edge(u1, u2) or G.has_edge(u1, v2) or 
                G.has_edge(v1, u2) or G.has_edge(v1, v2)):
                
                # Find intersection point
                intersection = geom1.intersection(geom2)
                
                # Only add if it's a proper intersection
                if not intersection.is_empty and intersection.geom_type == 'Point':
                    # Find the closest nodes on each route
                    dist1_u = Point(G.nodes[u1]['pos']).distance(intersection)
                    dist1_v = Point(G.nodes[v1]['pos']).distance(intersection)
                    dist2_u = Point(G.nodes[u2]['pos']).distance(intersection)
                    dist2_v = Point(G.nodes[v2]['pos']).distance(intersection)
                    
                    # Connect the closest nodes from each route
                    node1 = u1 if dist1_u < dist1_v else v1
                    node2 = u2 if dist2_u < dist2_v else v2
                    
                    G.add_edge(node1, node2,
                              road_name="intersection", 
                              road_type="I", 
                              weight=1.0)  # Short distance for intersection
                    route_intersections += 1
    
    print(f"Added {route_intersections} connections at route intersections")
    
    # Remove isolated nodes
    isolated = [node for node, degree in G.degree() if degree == 0]
    G.remove_nodes_from(isolated)
    print(f"Removed {len(isolated)} isolated nodes")
    
    # Print final graph statistics
    total_time = time.time() - start_time
    print(f"\nTotal graph creation time: {total_time:.4f} seconds")
    print(f"Final graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G

def visualize_graph(G, routes_gdf, intersections_gdf, output_path=None):
    """Visualize the topologically connected graph"""
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
        'I': 'orange',   # Intersection connection
        'J': 'cyan',     # Junction connection
        'U': 'gray'      # Unknown type
    }
    
    # Draw edges with appropriate colors
    edge_types_drawn = set()
    
    for u, v, data in G.edges(data=True):
        if u in pos and v in pos:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            road_type = data.get('road_type', 'U')
            color = road_colors.get(road_type, 'gray')
            
            # Draw with appropriate width and style
            linewidth = 2.0
            alpha = 0.8
            
            # Make connectors thinner
            if road_type in ['I', 'J']:
                linewidth = 1.0
                alpha = 0.6
            
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, 
                   alpha=alpha, label=f'Type {road_type}' if road_type not in edge_types_drawn else "")
            
            edge_types_drawn.add(road_type)
    
    # Separate nodes by type
    intersection_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'intersection']
    route_endpoint_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'route_endpoint']
    
    # Draw intersections in black
    if intersection_nodes:
        intersection_scatter = ax.scatter(
            [pos[n][0] for n in intersection_nodes if n in pos],
            [pos[n][1] for n in intersection_nodes if n in pos],
            s=25, c='black', alpha=0.8, label='Intersections'
        )
    
    # Draw route endpoints in light gray
    if route_endpoint_nodes:
        ax.scatter(
            [pos[n][0] for n in route_endpoint_nodes if n in pos],
            [pos[n][1] for n in route_endpoint_nodes if n in pos],
            s=15, c='lightgray', alpha=0.8, label='Route Endpoints'
        )
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
    
    # Set title and labels
    ax.set_title('Tampa Evacuation Network Graph (Topological Connections)')
    
    # Remove axes
    ax.set_axis_off()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to {output_path}")
    
    total_time = time.time() - start_time
    print(f"Visualization completed in {total_time:.4f} seconds")

def main():
    # Define data paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    routes_path = os.path.join(data_dir, 'Evacuation_Routes_Tampa.geojson')
    intersections_path = os.path.join(data_dir, 'Intersection.geojson')
    output_path = os.path.join(os.path.dirname(data_dir), 'evacuation_network.png')
    pickle_path = os.path.join(data_dir, 'evacuation_graph.pickle')
    
    start_time = time.time()
    
    print("Loading data...")
    routes_gdf, intersections_gdf = load_data(routes_path, intersections_path)
    loading_time = time.time() - start_time
    print(f"Data loading completed in {loading_time:.4f} seconds")
    
    print("Creating graph...")
    # Use build_topological_graph instead of create_graph
    G = build_topological_graph(routes_gdf, intersections_gdf)
    
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