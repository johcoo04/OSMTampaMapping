import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import os
import pickle
from shapely.geometry import Point, LineString
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
        
        # Get route endpoints from G
        start_node = f"r{nearest_route_idx}_start"
        end_node = f"r{nearest_route_idx}_end"
        
        # Connect to both endpoints of the road
        connections_added = 0
        for node in [start_node, end_node]:
            if node in G.nodes:  # Ensure the node exists
                node_pos = G.nodes[node]['pos']
                dist = Point(node_pos).distance(centroid_point)
                
                # Add edge with attributes
                G.add_edge(centroid_id, node,
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
                end_node_2 = f"r{second_nearest}_end"
                
                for node in [start_node_2, end_node_2]:
                    if node in G.nodes:
                        node_pos = G.nodes[node]['pos']
                        dist = Point(node_pos).distance(centroid_point)
                        G.add_edge(centroid_id, node,
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
    """Visualize the graph without showing nodes except centroids"""
    start_time = time.time()
    print("Starting visualization with matplotlib...")
    
    fig, ax = plt.subplots(figsize=(45, 45))
    
    # Extract node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Define colors for different road types (removed 'I')
    road_colors = {
        'C': 'blue',     # County road
        'H': 'red',      # Highway
        'M': 'green',    # Main road
        'P': 'purple',   # Primary/Peripheral road
        'J': 'cyan',     # Junction connection
        'Z': 'magenta',  # Centroid connector
        'U': 'gray'      # Unknown type
    }
    
    # Draw edges with appropriate colors
    edge_types_drawn = set()
    
    for u, v, data in G.edges(data=True):
        if u in pos and v in pos:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            road_type = data.get('road_type', 'U')
            # Skip type 'I' edges if any still exist in the graph
            if road_type == 'I':
                continue
                
            color = road_colors.get(road_type, 'gray')
            
            # Draw with appropriate width and style
            linewidth = 2.0
            alpha = 0.8

            if road_type in ['Z']:
                linewidth = 1.0
                alpha = 0.6
            
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, 
                   alpha=alpha, label=f'Type {road_type}' if road_type not in edge_types_drawn else "")
            
            edge_types_drawn.add(road_type)
    
    # COMMENTED OUT: Regular node visualization code
    """
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
            s=15, c='lightgray', alpha=0.6
        )
    """
    
    # Draw centroids (keep these visible)
    centroid_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'centroid']
    if centroid_nodes:
        centroid_scatter = ax.scatter(
            [pos[n][0] for n in centroid_nodes if n in pos],
            [pos[n][1] for n in centroid_nodes if n in pos],
            s=150, c='yellow', alpha=1.0, marker='*', edgecolor='black', label='Zip Code Centroids'
        )
    
    # Add legend with road types and centroids
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
    
    # Set title and labels
    ax.set_title('Tampa Evacuation Network Graph with Zip Code Centroids')
    
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
    import json
    import pandas as pd
    
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

def main():
    # Define data paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    routes_path = os.path.join(data_dir, 'Evacuation_Routes_Tampa.geojson')
    intersections_path = os.path.join(data_dir, 'Intersection.geojson')
    centroids_path = os.path.join(data_dir, 'zip_code_centroids.json')  # Path to zip code centroids
    output_path = os.path.join(os.path.dirname(data_dir), 'evacuation_network.png')
    pickle_path = os.path.join(data_dir, 'evacuation_graph.pickle')
    
    start_time = time.time()
    
    print("Loading data...")
    routes_gdf, intersections_gdf = load_data(routes_path, intersections_path)
    loading_time = time.time() - start_time
    print(f"Data loading completed in {loading_time:.4f} seconds")
    
    # Load zip code centroids
    print("Loading zip code centroids...")
    centroids_gdf = load_centroids(centroids_path)
    print(f"Loaded {len(centroids_gdf)} zip code centroids")
    
    print("Creating graph...")
    # Use build_topological_graph instead of create_graph
    G = build_topological_graph(routes_gdf, intersections_gdf)
    
    # Add centroids to the graph
    print("Adding zip code centroids to graph...")
    G = add_centroids_to_graph(G, centroids_gdf, routes_gdf)
    
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