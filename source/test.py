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

def visualize_graph(G, routes_gdf, intersections_gdf, output_path=None):
    """Visualize the graph with matplotlib using NetworkX drawing functions"""
    start_time = time.time()
    print("Starting visualization...")
    
    fig, ax = plt.subplots(figsize=(45, 45))
    
    # Extract node positions from graph attributes
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw the graph edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='blue', width=1.0, alpha=0.7)
    print(nx)
    
    # Draw the graph nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=15, node_color='black', alpha=0.8)
    
    # Optionally draw labels (might be cluttered with many nodes)
    # nx.draw_networkx_labels(G, pos, font_size=8, alpha=0.7)
    
    # Set title and labels
    ax.set_title('Tampa Evacuation Network Graph')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    # Remove axes
    ax.set_axis_off()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to {output_path}")
    
    plt.close()
    
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
    
    # Check if pickle file exists
    if os.path.exists(pickle_path):
        print(f"Loading graph from pickle file: {pickle_path}")
        with open(pickle_path, 'rb') as f:
            G = pickle.load(f)
        print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        loading_time = time.time() - start_time
        print(f"Graph loading completed in {loading_time:.4f} seconds")
        
        # Load the geodataframes for visualization
        print("Loading data for visualization...")
        routes_gdf, intersections_gdf = load_data(routes_path, intersections_path)
    else:
        print("Loading data...")
        routes_gdf, intersections_gdf = load_data(routes_path, intersections_path)
        loading_time = time.time() - start_time
        print(f"Data loading completed in {loading_time:.4f} seconds")
        
        print("Creating graph...")
        G = create_graph(routes_gdf, intersections_gdf)
        
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