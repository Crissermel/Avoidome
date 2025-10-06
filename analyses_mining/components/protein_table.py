import streamlit as st
import pandas as pd

def show_protein_table(df, show_stats=True):
    st.write("## Protein Table")
    st.dataframe(df)
    if show_stats:
        st.write("## Summary Stats")
        st.write(df.describe(include='all')) 