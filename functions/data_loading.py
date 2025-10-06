# functions/data_loading.py
import pandas as pd
from pathlib import Path
import streamlit as st

def load_csv(filepath, **kwargs):
    """Generic CSV loader."""
    return pd.read_csv(filepath, **kwargs)

# Add more specific data loading functions as needed, e.g.:
def load_avoidome_prot_list(filepath):
    return pd.read_csv(filepath)

def load_enriched_prot_list(filepath):
    return pd.read_csv(filepath)

def load_bioactivity_profile(csv_path=None):
    """Load the bioactivity profile CSV, return DataFrame or None if not found."""
    from pathlib import Path
    import streamlit as st
    if csv_path is None:
        csv_path = Path(__file__).parent.parent / "processed_data/avoidome_bioactivity_profile.csv"
    else:
        csv_path = Path(csv_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        st.warning(f"CSV file not found: {csv_path.name}")
        return None

def load_chembl_mapped_bioactivity_profile(csv_path=None):
    """Load the ChEMBL-mapped bioactivity profile CSV, return DataFrame or None if not found."""
    from pathlib import Path
    import streamlit as st
    if csv_path is None:
        csv_path = Path(__file__).parent.parent / "processed_data/chembl_mapped_bioactivity_profile.csv"
    else:
        csv_path = Path(csv_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        st.warning(f"CSV file not found: {csv_path.name}")
        return None 