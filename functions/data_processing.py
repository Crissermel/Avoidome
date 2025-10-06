# functions/data_processing.py

def filter_by_column(df, column, value):
    """Filter DataFrame by column value."""
    return df[df[column] == value]

# Add more data processing functions as needed

def enrich_with_function_and_pathways(df, enrichment_df):
    """Merge main df with enrichment info on UniProt ID."""
    return df.merge(enrichment_df, on='UniProt ID', how='left')

# ...add other data processing functions as needed 