#!/usr/bin/env python3
"""
Unified Dashboard Runner

This script launches the unified Avoidome QSAR dashboard.

Usage:
    python run_unified_dashboard.py
"""

import subprocess
import sys
import os

def main():
    """Run the unified dashboard"""
    print("Starting Unified Avoidome QSAR Dashboard...")
    print("Dashboard will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    try:
        # Run the unified dashboard
        subprocess.run([
            "streamlit", "run", "unified_dashboard.py",
            "--server.port", "8501",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")
    except FileNotFoundError:
        print("Error: streamlit not found. Please install streamlit:")
        print("pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"Error running dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 