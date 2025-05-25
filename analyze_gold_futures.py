#!/usr/bin/env python3
"""
Gold Futures Market Analysis Script
--------------------------------------
This script runs the complete market analysis for Gold Futures (GC=F),
generating reports and visualizations for different timeframes.
"""

import os
import sys
import subprocess
from datetime import datetime

# Run the analysis on Gold Futures (GC=F)
def analyze_gold_futures():
    """Run Gold Futures market analysis using the main analysis script."""
    print(f"Starting Gold Futures market analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Using the multi-market run_analysis.py script with commodity parameters
    cmd = [
        sys.executable,
        "run_analysis.py",
        "GC=F",
        "--market-type", "commodity",
        "--timeframes", "1d", "1w",
        "--source", "yahoo_finance"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("Gold Futures analysis completed successfully!")
        
        # Print the location of the results
        results_dir = os.path.join("output", "commodity", "GC=F")
        if os.path.exists(results_dir):
            print(f"\nResults are available in: {os.path.abspath(results_dir)}")
            
            # List the generated files
            print("\nGenerated files:")
            for root, dirs, files in os.walk(results_dir):
                rel_path = os.path.relpath(root, results_dir)
                if rel_path == ".":
                    rel_path = ""
                for file in files:
                    print(f"  {os.path.join(rel_path, file)}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running Gold Futures analysis: {e}")
        return False

if __name__ == "__main__":
    analyze_gold_futures() 