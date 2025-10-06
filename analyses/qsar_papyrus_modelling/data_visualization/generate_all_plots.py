#!/usr/bin/env python3
"""
Generate All Visualization Plots for Papyrus QSAR Dashboard
This script generates all the plots needed for the dashboard.
"""

import sys
import os
sys.path.append('/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling')

from data_visualization.visualization_utils import generate_all_visualizations

if __name__ == "__main__":
    print("Generating all visualization plots for Papyrus QSAR Dashboard...")
    stats = generate_all_visualizations()
    
    if stats:
        print("\nVisualization Summary:")
        print(f"- Bioactivity overview plots generated")
        print(f"- Bioactivity points plots generated")
        print(f"- QSAR performance plots generated")
        print(f"- Protein detail plots generated for top proteins")
        if 'classification_stats' in stats:
            print(f"- Classification performance plots generated")
            print(f"- Classification detailed analysis generated")
        print("\nAll plots saved to: analyses/qsar_papyrus_modelling/data_visualization/plots/")
    else:
        print("Error: Could not generate visualizations. Check if data files exist.") 