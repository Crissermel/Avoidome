#!/usr/bin/env python3
"""
Simple script to view QSAR model plots

This script provides a simple interface to view the generated plots.
"""

import os
import webbrowser
from pathlib import Path

def main():
    """Main function to display plot information"""
    plots_dir = Path("analyses/standardized_qsar_models/plots")
    
    print("QSAR Model Results - Generated Plots")
    print("=" * 50)
    print(f"Plot directory: {plots_dir}")
    print()
    
    # List all generated plots
    png_files = list(plots_dir.glob("*.png"))
    html_files = list(plots_dir.glob("*.html"))
    
    print("Static Plots (PNG files):")
    print("-" * 30)
    for i, png_file in enumerate(png_files, 1):
        size_mb = png_file.stat().st_size / (1024 * 1024)
        print(f"{i:2d}. {png_file.name} ({size_mb:.1f} MB)")
    
    print()
    print("Interactive Plots (HTML files):")
    print("-" * 35)
    for i, html_file in enumerate(html_files, 1):
        size_mb = html_file.stat().st_size / (1024 * 1024)
        print(f"{i:2d}. {html_file.name} ({size_mb:.1f} MB)")
    
    print()
    print("Summary Report:")
    print("-" * 15)
    report_file = plots_dir / "plotting_report.md"
    if report_file.exists():
        print(f"plotting_report.md ({report_file.stat().st_size} bytes)")
    
    print()
    print("To view plots:")
    print("- Open PNG files with any image viewer")
    print("- Open HTML files in a web browser")
    print("- View the summary report for detailed information")
    
    print()
    print("Key Insights:")
    print("-" * 15)
    print("• 88 completed models across 4 types and 3 organisms")
    print("• Human models dominate (54 models, 61.4%)")
    print("• Average R² score: 0.571 ± 0.099")
    print("• Average accuracy: 0.836 ± 0.071")
    print("• Best performing protein: CHRM1 (R² = 0.781)")
    print("• Best classification: SCN5A (Accuracy = 0.988)")

if __name__ == "__main__":
    main()