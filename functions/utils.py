# functions/utils.py

def get_unique_values(df, column):
    """Return unique values from a DataFrame column."""
    return df[column].unique().tolist()

# Add more utility/helper functions as needed 