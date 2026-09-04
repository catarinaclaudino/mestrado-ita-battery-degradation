"""
================================================================================
STEP 1: DATASET INSPECTION
================================================================================

This script inspects the CALCE CS2-35 dataset structure without making
assumptions about its organization.

Purpose:
- Identify file names and types
- Determine dataframe dimensions
- Extract column names and data types
- Check for missing values, duplicates, anomalies
- Create a data dictionary
- Report exact dataset structure before proceeding

IMPORTANT:
Do NOT assume column names or file structure before inspection.
Do NOT make any preprocessing decisions yet.

================================================================================
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = r"C:\Users\catar\OneDrive\Área de Trabalho\MESTRADO\TESTE_METODOLOGIA\CS2_35"
OUTPUT_DIR = r"C:\Users\catar\OneDrive\Área de Trabalho\MESTRADO\TESTE_METODOLOGIA"

# ============================================================================
# STEP 1A: LIST ALL FILES IN THE DATASET DIRECTORY
# ============================================================================

print("\n" + "="*80)
print("STEP 1: DATASET INSPECTION - CALCE CS2-35")
print("="*80)

print(f"\nDataset Path: {DATASET_PATH}")
print(f"Output Directory: {OUTPUT_DIR}")

# Check if directory exists
if not os.path.exists(DATASET_PATH):
    print(f"\n[ERROR] Dataset directory not found: {DATASET_PATH}")
    exit(1)

print("\n" + "-"*80)
print("1A: FILES IN DATASET DIRECTORY")
print("-"*80)

all_files = os.listdir(DATASET_PATH)
print(f"\nTotal files/folders: {len(all_files)}")
print("\nContents:")
for item in sorted(all_files):
    item_path = os.path.join(DATASET_PATH, item)
    if os.path.isdir(item_path):
        print(f"  [DIR]  {item}")
    else:
        file_size = os.path.getsize(item_path) / (1024 * 1024)  # MB
        print(f"  [FILE] {item} ({file_size:.2f} MB)")

# ============================================================================
# STEP 1B: IDENTIFY DATA FILES
# ============================================================================

print("\n" + "-"*80)
print("1B: IDENTIFYING DATA FILES")
print("-"*80)

data_files = {}
for item in all_files:
    item_path = os.path.join(DATASET_PATH, item)
    if not os.path.isdir(item_path):
        if item.endswith(('.csv', '.xlsx', '.xls', '.txt', '.json')):
            data_files[item] = item_path
            print(f"  Found: {item}")

if not data_files:
    print("  [WARNING] No standard data files found (.csv, .xlsx, .xls, .txt, .json)")

# ============================================================================
# STEP 1C: LOAD AND INSPECT EACH DATA FILE
# ============================================================================

print("\n" + "-"*80)
print("1C: LOADING AND INSPECTING DATA FILES")
print("-"*80)

dataframes = {}

for filename, filepath in data_files.items():
    print(f"\n--- File: {filename} ---")
    
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filepath)
        elif filename.endswith('.txt'):
            # Try comma-separated first
            try:
                df = pd.read_csv(filepath)
            except:
                # Try space-separated
                df = pd.read_csv(filepath, sep='\s+')
        elif filename.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            print("  [SKIP] Unrecognized format")
            continue
        
        dataframes[filename] = df
        
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Data types:\n{df.dtypes}")
        print(f"  First few rows:")
        print(df.head(10))
        print(f"  Missing values:\n{df.isnull().sum()}")
        print(f"  Duplicated rows: {df.duplicated().sum()}")
        
    except Exception as e:
        print(f"  [ERROR] Could not load: {e}")

# ============================================================================
# STEP 1D: DIRECTORY STRUCTURE (IF MULTIPLE BATTERY FILES)
# ============================================================================

print("\n" + "-"*80)
print("1D: CHECKING FOR BATTERY-SPECIFIC SUBDIRECTORIES")
print("-"*80)

battery_dirs = {}
for item in all_files:
    item_path = os.path.join(DATASET_PATH, item)
    if os.path.isdir(item_path):
        contents = os.listdir(item_path)
        battery_files = [f for f in contents if f.endswith(('.csv', '.xlsx', '.xls', '.txt', '.json'))]
        if battery_files:
            battery_dirs[item] = battery_files
            print(f"\n{item}:")
            print(f"  Files: {battery_files}")

# ============================================================================
# STEP 1E: SUMMARY OF FINDINGS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY OF DATASET STRUCTURE")
print("="*80)

print(f"\n✓ Standalone data files found: {len(dataframes)}")
for fname in dataframes.keys():
    print(f"  - {fname}")

print(f"\n✓ Battery-specific directories found: {len(battery_dirs)}")
for bdir, files in battery_dirs.items():
    print(f"  - {bdir}: {len(files)} file(s)")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)

if len(dataframes) > 0:
    print("\n[1] Standalone files will be analyzed")
    print("[2] Determine which columns contain:")
    print("    - Battery identifier")
    print("    - Cycle number")
    print("    - Discharge capacity")
    print("    - Operating conditions (if present)")
    print("    - Timestamps (if present)")

if len(battery_dirs) > 0:
    print("\n[1] Battery-specific structure detected")
    print("[2] Each battery may have separate files")
    print("[3] Will need to load and consolidate across batteries")

print("\n" + "="*80)
print("Inspection complete. Review output above before proceeding.")
print("="*80 + "\n")
