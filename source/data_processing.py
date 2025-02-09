import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import unary_union

def calculate_min_distances(centroids_gdf, routes_gdf):
    min_distances = []
    lines = []
    
    for centroid in centroids_gdf.itertuples():
        distances = routes_gdf.distance(centroid.geometry)
        min_distance = distances.min()
        nearest_route = routes_gdf.iloc[distances.idxmin()]
        nearest_point = nearest_route.geometry.interpolate(nearest_route.geometry.project(centroid.geometry))
        min_distances.append({'ZipCode': centroid.ZipCode, 'min_distance': min_distance, 'nearest_point': nearest_point})
        lines.append(LineString([centroid.geometry, nearest_point]))
    
    return min_distances, lines

def filter_ramps_within_distance(ramps_gdf, routes_gdf, distance=500):
    filtered_ramps = ramps_gdf[ramps_gdf.geometry.apply(lambda ramp: routes_gdf.distance(ramp).min() <= distance)]
    return filtered_ramps

def merge_close_nodes(nodes_gdf, distance=50):
    merged_nodes = []
    while not nodes_gdf.empty:
        node = nodes_gdf.iloc[0]
        close_nodes = nodes_gdf[nodes_gdf.distance(node.geometry) <= distance]
        centroid = unary_union(close_nodes.geometry).centroid
        merged_nodes.append({'geometry': centroid})
        nodes_gdf = nodes_gdf.drop(close_nodes.index)
    return gpd.GeoDataFrame(merged_nodes, crs=nodes_gdf.crs)