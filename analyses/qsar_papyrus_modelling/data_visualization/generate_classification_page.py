#!/usr/bin/env python3
"""
Generate Classification Page for Papyrus QSAR Dashboard
This script generates classification plots to add as a page to the existing dashboard.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add the parent directory to the path
sys.path.append('/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling')

from data_visualization.visualization_utils import PapyrusVisualizer

def generate_classification_page():
    """Generate classification page for the dashboard"""
    
    print("Generating Classification Page for Papyrus QSAR Dashboard...")
    
    # Initialize visualizer
    visualizer = PapyrusVisualizer()
    
    # Load classification results
    classification_path = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling/classification_results.csv"
    
    if not os.path.exists(classification_path):
        print(f"Classification results not found at: {classification_path}")
        return None
    
    classification_df = pd.read_csv(classification_path)
    print(f"Loaded classification results: {len(classification_df)} total records")
    
    # Filter successful models
    successful_models = classification_df[classification_df['accuracy'].notna()]
    print(f"Successful models: {len(successful_models)} records")
    
    if len(successful_models) == 0:
        print("No successful classification models found!")
        return None
    
    # Generate classification performance plots
    print("Generating classification performance plots...")
    classification_stats = visualizer.generate_classification_performance_plots(classification_df)
    
    # Generate detailed analysis
    print("Generating classification detailed analysis...")
    classification_detailed_stats = visualizer.generate_classification_detailed_analysis(classification_df)
    
    print(f"Classification page generated successfully!")
    print(f"All plots saved to: {visualizer.output_dir}")
    
    return {
        'classification_stats': classification_stats,
        'classification_detailed_stats': classification_detailed_stats
    }

if __name__ == "__main__":
    stats = generate_classification_page()
    
    if stats:
        print("\nClassification Page Summary:")
        print(f"- Classification performance plots generated")
        print(f"- Classification detailed analysis generated")
        print("\nClassification page added to dashboard!")
    else:
        print("Error: Could not generate classification page. Check if classification results exist.") 