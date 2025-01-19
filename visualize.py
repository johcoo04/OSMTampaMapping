import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.patches as mpatches

def plot_population_and_routes(zip_gdf, routes_gdf):
    """
    Plot population by ZIP code and overlay evacuation routes.

    Args:
        zip_gdf (gpd.GeoDataFrame): GeoDataFrame with ZIP code and population data.
        routes_gdf (gpd.GeoDataFrame): GeoDataFrame with evacuation routes.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Plot ZIP codes with population data
    zip_gdf.plot(column='Population', cmap='OrRd', legend=True, ax=ax, missing_kwds={
        "color": "lightgrey",
        "label": "No Data"
    })
    
    # Define a color map for road types
    road_type_colors = {
        'C': 'green',    # County road
        'H': 'blue',     # Highway
        'M': 'orange',   # Main road
        'P': 'purple'    # Peripheral or Primary road
    }
    
    # Overlay evacuation routes with color coding based on road type
    for road_type, color in road_type_colors.items():
        subset = routes_gdf[routes_gdf['TYPE'] == road_type]
        subset.plot(ax=ax, color=color, linewidth=1, label=road_type)
    
    # Create custom legend handles
    legend_handles = [mpatches.Patch(color=color, label=road_type) for road_type, color in road_type_colors.items()]
    
    plt.title("Population by ZIP Code with Evacuation Routes")
    plt.legend(handles=legend_handles, title="Road Types")
    
    # Save the plot to a file
    plt.savefig('colorize_routes.png')
    plt.show()
