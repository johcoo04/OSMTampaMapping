import pandas as pd

def load_population_data(json_path):
    """
    Load and prepare population data from a JSON file.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        pd.DataFrame: A DataFrame with population data.
    """
    population_df = pd.read_json(json_path)
    population_df.rename(columns={'zip': 'ZipCode', 'population': 'Population'}, inplace=True)
    population_df['ZipCode'] = population_df['ZipCode'].astype(str)
    return population_df
