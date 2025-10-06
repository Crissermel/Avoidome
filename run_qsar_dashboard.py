#!/usr/bin/env python3
"""
Runner script for QSAR Modeling Dashboard

This script launches the QSAR modeling dashboard using Streamlit.

Usage:
    python run_qsar_dashboard.py
    # or
    streamlit run qsar_modeling_dashboard.py
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas',
        'matplotlib',
        'seaborn',
        'numpy',
        'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        "analyses/standardized_qsar_models/modeling_summary.csv",
        "analyses/standardized_qsar_models/esm_morgan_modeling_summary.csv"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Missing required data files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease ensure the QSAR models have been trained and the summary files exist.")
        return False
    
    return True

def main():
    """Main function to run the dashboard"""
    print("QSAR Modeling Dashboard Launcher")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check data files
    if not check_data_files():
        sys.exit(1)
    
    print("All requirements satisfied")
    print("Launching QSAR Modeling Dashboard...")
    print("\nThe dashboard will open in your default web browser.")
    print("Press Ctrl+C to stop the dashboard.")
    print("=" * 40)
    
    try:
        # Launch Streamlit dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "qsar_modeling_dashboard.py",
            "--server.port", "8502",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()