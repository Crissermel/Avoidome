import sys
import os
components_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'components'))
if components_path not in sys.path:
    sys.path.insert(0, components_path)
from protein_table import show_protein_table

import streamlit as st
import pandas as pd

st.title("Avoidome Protein List Visualization")

csv_path = os.path.join(os.path.dirname(__file__), '../primary_data/avoidome_prot_list.csv')

def load_data():
    return pd.read_csv(csv_path)

def main():
    df = load_data()
    show_protein_table(df)

if __name__ == "__main__":
    main() 