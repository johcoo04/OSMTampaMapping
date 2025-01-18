import logging
import pandas as pd

from loggingFormatterEST import setup_logging

# Configure logging
logger = setup_logging('inspect_csv.log','inspect_csv')

# Load the CSV file
def inspect_csv(file_path):

    df = pd.read_csv(file_path)
    # Display the first few rows of the dataframe
    logger(df.head())

    # Display the number of rows and columns
    logger(f"Number of rows: {df.shape[0]}")
    logger(f"Number of columns: {df.shape[1]}")

    # Display the count of each type
    logger(df['type'].value_counts())