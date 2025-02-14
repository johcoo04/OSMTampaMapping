import geopandas as gpd
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection
from shapely.ops import unary_union

def filter_ramps_near_highways(ramps_gdf, routes_gdf, distance=500):
    highways_gdf = routes_gdf[routes_gdf['TYPE'] == 'H']
    filtered_ramps = ramps_gdf[ramps_gdf.geometry.apply(lambda ramp: highways_gdf.distance(ramp).min() <= distance)]
    return filtered_ramps

def calculate_min_distances_to_routes(centroids_gdf, routes_gdf):
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

def calculate_min_distances_to_ramps(centroids_gdf, ramps_gdf, num_paths=3):
    min_distances = []
    lines = []
    for centroid in centroids_gdf.itertuples():
        distances = ramps_gdf.distance(centroid.geometry)
        nearest_ramps = distances.nsmallest(num_paths).index
        for idx in nearest_ramps:
            nearest_ramp = ramps_gdf.loc[idx]
            nearest_point = nearest_ramp.geometry
            min_distance = distances[idx]
            min_distances.append({'ZipCode': centroid.ZipCode, 'min_distance': min_distance, 'nearest_point': nearest_point})
            lines.append(LineString([centroid.geometry, nearest_point]))
    return min_distances, lines

def filter_intersections_near_route(intersections_gdf, routes_gdf, distance=500):
    filtered_intersections = intersections_gdf[intersections_gdf.geometry.apply(lambda intersection: routes_gdf.distance(intersection).min() <= distance)]
    return filtered_intersections

def merge_close_nodes(nodes_gdf, distance=50):
    merged_nodes = []
    while not nodes_gdf.empty:
        node = nodes_gdf.iloc[0]
        close_nodes = nodes_gdf[nodes_gdf.distance(node.geometry) <= distance]
        centroid = unary_union(close_nodes.geometry).centroid
        merged_nodes.append({'geometry': centroid})
        nodes_gdf = nodes_gdf.drop(close_nodes.index)
    return gpd.GeoDataFrame(merged_nodes, crs=nodes_gdf.crs)

def create_nodes_along_routes(routes_gdf, interval=500):
    nodes = []
    for route in routes_gdf.itertuples():
        length = route.geometry.length
        num_points = int(length // interval)
        for i in range(num_points + 1):
            point = route.geometry.interpolate(i * interval)
            nodes.append({'geometry': point})
    return gpd.GeoDataFrame(nodes, crs=routes_gdf.crs)

# def create_intersection_nodes(routes_gdf):
#     intersections = []
#     for i, route1 in enumerate(routes_gdf.geometry):
#         for j, route2 in enumerate(routes_gdf.geometry):
#             if i >= j:
#                 continue
#             intersection = route1.intersection(route2)
#             if not intersection.is_empty:
#                 if intersection.geom_type == 'Point':
#                     intersections.append({'geometry': intersection})
#                 elif intersection.geom_type == 'MultiPoint':
#                     for point in intersection.geoms:
#                         intersections.append({'geometry': point})
#                 elif intersection.geom_type == 'GeometryCollection':
#                     for geom in intersection.geoms:
#                         if geom.geom_type == 'Point':
#                             intersections.append({'geometry': geom})
#     return gpd.GeoDataFrame(intersections, crs=routes_gdf.crs)