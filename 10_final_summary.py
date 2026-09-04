"""
================================================================================
FINAL COMPREHENSIVE SUMMARY
================================================================================

This script generates a comprehensive summary of the analysis results.

================================================================================
"""

import os
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TABLES_DIR, FIGURES_DIR, DATA_DIR, OUTPUT_BASE_DIR, VERBOSE, DIVIDER_WIDTH

def print_section(title):
    if VERBOSE:
        print(f"\n{'='*DIVIDER_WIDTH}")
        print(title)
        print('='*DIVIDER_WIDTH)

def main():
    print_section("ANALYSIS SUMMARY REPORT")
    
    print(f"\nAnalysis Output Location: {OUTPUT_BASE_DIR}")
    
    # List generated tables
    print_section("GENERATED TABLES")
    if os.path.exists(TABLES_DIR):
        tables = sorted([f for f in os.listdir(TABLES_DIR) if f.endswith('.csv')])
        print(f"\nTotal tables generated: {len(tables)}\n")
        for i, table in enumerate(tables, 1):
            table_path = os.path.join(TABLES_DIR, table)
            size = os.path.getsize(table_path) / 1024  # KB
            print(f"{i:2d}. {table:50s} ({size:8.2f} KB)")
    
    # List generated figures
    print_section("GENERATED FIGURES")
    if os.path.exists(FIGURES_DIR):
        figures = sorted([f for f in os.listdir(FIGURES_DIR) if f.endswith('.png')])
        print(f"\nTotal figures generated: {len(figures)}\n")
        for i, fig in enumerate(figures, 1):
            fig_path = os.path.join(FIGURES_DIR, fig)
            size = os.path.getsize(fig_path) / (1024 * 1024)  # MB
            print(f"{i:2d}. {fig:50s} ({size:8.2f} MB)")
    
    # Summary of key findings
    print_section("KEY RESULTS SUMMARY")
    
    # Read and display summary tables if they exist
    summary_files = [
        ('Table_01_SoH_Summary.csv', 'State of Health Summary'),
        ('Table_05_Model_Comparison.csv', 'Model Comparison'),
        ('Table_06_Shapiro_Wilk_Normality_Test.csv', 'Residual Normality Test'),
        ('Table_09_Pettitt_Change_Point_Detection.csv', 'Pettitt Change-Point Test'),
        ('Table_10_Failure_and_RUL_Analysis.csv', 'Failure and RUL Analysis')
    ]
    
    for filename, title in summary_files:
        filepath = os.path.join(TABLES_DIR, filename)
        if os.path.exists(filepath):
            print(f"\n{title}:")
            print("-" * DIVIDER_WIDTH)
            try:
                df = pd.read_csv(filepath)
                print(df.to_string(index=False))
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    print_section("ANALYSIS COMPLETE")
    print(f"\n✓ Analysis pipeline executed successfully")
    print(f"✓ All results saved to: {OUTPUT_BASE_DIR}")
    print(f"\nNext steps:")
    print(f"  1. Review generated figures in: {FIGURES_DIR}")
    print(f"  2. Review generated tables in: {TABLES_DIR}")
    print(f"  3. Review processed data in: {DATA_DIR}")
    print(f"\n{'='*DIVIDER_WIDTH}\n")

if __name__ == "__main__":
    main()
