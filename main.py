from process_geojson import load_zip_geojson, load_routes_geojson, generate_zip_code_nodes
from process_graph import combine_data_to_graph
from visualize import visualize_combined_graph

# File paths
zip_geojson_path = "data/Zip_Codes_Tampa.geojson"
routes_geojson_path = "data/Evacuation_Routes_Tampa.geojson"


# Load data
zip_codes_gdf = load_zip_geojson(zip_geojson_path, zip_column="Zip_Code")
routes_gdf = load_routes_geojson(routes_geojson_path)
# shelters_gdf = load_shelters_geojson(shelters_geojson_path, shelter_id_column="ID") -- If want to use shelters later

# Combine ZIP nodes with routes
combined_graph = combine_data_to_graph(zip_codes_gdf, routes_gdf)

# Visualize the graph
visualize_combined_graph(combined_graph)
