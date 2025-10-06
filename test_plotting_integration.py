#!/usr/bin/env python3
"""
Test script to verify plotting integration in dashboard
"""

import os
from pathlib import Path

def test_plotting_integration():
    """Test if plotting integration works correctly"""
    print("Testing QSAR Dashboard Plotting Integration")
    print("=" * 50)
    
    # Check if plots directory exists
    plots_dir = Path("analyses/standardized_qsar_models/plots")
    if not plots_dir.exists():
        print("ERROR: Plots directory not found")
        print("Run: python plot_qsar_results.py")
        return False
    
    # Check for PNG files
    png_files = list(plots_dir.glob("*.png"))
    print(f"Static plots found: {len(png_files)}")
    for png_file in png_files:
        size_mb = png_file.stat().st_size / (1024 * 1024)
        print(f"  - {png_file.name} ({size_mb:.1f} MB)")
    
    # Check for HTML files
    html_files = list(plots_dir.glob("*.html"))
    print(f"Interactive plots found: {len(html_files)}")
    for html_file in html_files:
        size_mb = html_file.stat().st_size / (1024 * 1024)
        print(f"  - {html_file.name} ({size_mb:.1f} MB)")
    
    # Check for report
    report_file = plots_dir / "plotting_report.md"
    if report_file.exists():
        print(f"Report found: {report_file.name} ({report_file.stat().st_size} bytes)")
    
    # Test dashboard plotting functions
    print("\nTesting dashboard plotting functions...")
    
    try:
        import sys
        sys.path.append('.')
        from qsar_modeling_dashboard import get_available_plots
        
        png_files, html_files = get_available_plots()
        print(f"Dashboard detected {len(png_files)} static plots")
        print(f"Dashboard detected {len(html_files)} interactive plots")
        
        # List detected plots
        print("\nDetected static plots:")
        for filename, description in png_files.items():
            print(f"  - {filename}: {description}")
        
        print("\nDetected interactive plots:")
        for filename, description in html_files.items():
            print(f"  - {filename}: {description}")
        
        print("\nSUCCESS: Plotting integration working correctly!")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to test dashboard integration: {e}")
        return False

if __name__ == "__main__":
    success = test_plotting_integration()
    if success:
        print("\nDashboard plotting integration is ready!")
        print("Access the dashboard at: http://localhost:8502")
        print("Navigate to 'Visualizations' section to view plots")
    else:
        print("\nPlease fix the issues above before using the dashboard")