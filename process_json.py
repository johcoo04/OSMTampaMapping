import pandas as pd

# Load and prepare population data from a JSON file.
def load_population_data(json_path):
    population_df = pd.read_json(json_path)
    population_df.rename(columns={'zip': 'ZipCode', 'population': 'Population'}, inplace=True)
    population_df['ZipCode'] = population_df['ZipCode'].astype(str)
    return population_df
