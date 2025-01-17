import matplotlib.pyplot as plt
import geopandas as gpd

def plot_population_and_routes(zip_gdf, routes_gdf, output_file=None):
    """
    Plot population by ZIP code and overlay evacuation routes.

    Args:
        zip_gdf (gpd.GeoDataFrame): GeoDataFrame with ZIP code and population data.
        routes_gdf (gpd.GeoDataFrame): GeoDataFrame with evacuation routes.
        output_file (str): Optional. Path to save the plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Plot ZIP codes with population data
    zip_gdf.plot(column='Population', cmap='OrRd', legend=True, ax=ax, missing_kwds={
        "color": "lightgrey",
        "label": "No Data"
    })
    
    # Overlay evacuation routes
    routes_gdf.plot(ax=ax, color='blue', linewidth=1, label="Evacuation Routes")
    
    plt.title("Population by ZIP Code with Evacuation Routes")
    plt.legend()
    
    if output_file:
        plt.savefig(output_file, dpi=300)
    plt.savefig('output.png')
    
    #plt.show()
