import networkx as nx
import geopandas as gpd

# Define road type attributes
road_attributes = {
    'H': {'speed_limit': 100, 'capacity': 2000},  # Highway
    'P': {'speed_limit': 60, 'capacity': 1000},   # Primary road
    'C': {'speed_limit': 40, 'capacity': 500},    # Collector road
    'M': {'speed_limit': 30, 'capacity': 300}     # Minor road
}

def create_graph(centroids_gdf, routes_gdf, merged_nodes_gdf, route_lines_gdf, ramp_lines_gdf, trimmed_intersections_gdf):
    G = nx.DiGraph()  # Directed graph

    # Add nodes for centroids (Origin Nodes)
    # for idx, row in centroids_gdf.iterrows():
    #     G.add_node(row['ZipCode'], pos=(row.geometry.x, row.geometry.y), type='origin')

    # Add nodes for intersections (Intersection Nodes)
    for idx, row in trimmed_intersections_gdf.iterrows():
        G.add_node(f"intersection_{idx}", pos=(row.geometry.x, row.geometry.y), type='intersection')

    # Add nodes for merged ramps (Destination Nodes)
    # for idx, row in merged_nodes_gdf.iterrows():
    #     G.add_node(f"ramp_{idx}", pos=(row.geometry.x, row.geometry.y), type='destination')

    # # Add edges for the shortest paths to routes
    print(route_lines_gdf.head())
    for idx, line in route_lines_gdf.iterrows():
        print(line)
        # start = (line['geometry'].coords[0][0], line['geometry'].coords[0][1])
        # end = (line['geometry'].coords[1][0], line['geometry'].coords[1][1])
        # road_type = line['road_type']  # Assuming 'road_type' column exists
        # if road_type not in road_attributes:
        #     raise ValueError(f"Invalid road type: {road_type}")
        # attributes = road_attributes[road_type]
        # travel_time = line['geometry'].length / attributes['speed_limit']
        # G.add_edge(start, end, weight=travel_time, capacity=attributes['capacity'], distance=line['geometry'].length, road_type=road_type)

    # # Add edges for the shortest paths to ramps
    # for idx, line in ramp_lines_gdf.iterrows():
    #     start = (line['geometry'].coords[0][0], line['geometry'].coords[0][1])
    #     end = (line['geometry'].coords[1][0], line['geometry'].coords[1][1])
    #     road_type = line['road_type']  # Assuming 'road_type' column exists
    #     if road_type not in road_attributes:
    #         raise ValueError(f"Invalid road type: {road_type}")
    #     attributes = road_attributes[road_type]
    #     travel_time = line['geometry'].length / attributes['speed_limit']
    #     G.add_edge(start, end, weight=travel_time, capacity=attributes['capacity'], distance=line['geometry'].length, road_type=road_type)

    return G