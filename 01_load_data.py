"""
================================================================================
STEP 2: DATA LOADING
================================================================================

This script loads the CALCE CS2-35 dataset from files into memory.

Purpose:
- Load all data files from the dataset directory
- Handle multiple file formats (CSV, Excel, etc.)
- Consolidate data if distributed across multiple files
- Preserve data integrity and chronological order
- Create a unified dataframe for processing

IMPORTANT:
- Do NOT modify data at this stage
- Do NOT remove observations
- Do NOT assume column names
- Preserve all available information

================================================================================
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATASET_PATH, OUTPUT_BASE_DIR, DATA_DIR, VERBOSE

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_section(title):
    """Print a formatted section header."""
    if VERBOSE:
        print(f"\n{'='*80}")
        print(title)
        print('='*80)

def print_subsection(title):
    """Print a formatted subsection header."""
    if VERBOSE:
        print(f"\n{'-'*80}")
        print(title)
        print('-'*80)

# ============================================================================
# LOAD DATA FROM DIRECTORY
# ============================================================================

def load_dataset():
    """
    Load all data files from the CALCE dataset directory.
    
    Returns:
    --------
    dict
        Dictionary with filenames as keys and loaded dataframes as values.
    
    Raises:
    -------
    FileNotFoundError
        If dataset directory does not exist.
    ValueError
        If no data files are found.
    """
    
    print_section("STEP 2: DATA LOADING - CALCE CS2-35")
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_PATH}")
    
    print(f"Loading from: {DATASET_PATH}\n")
    
    # List all files in directory
    all_items = os.listdir(DATASET_PATH)
    
    # Identify data files
    data_files = {}
    subdirs = {}
    
    for item in sorted(all_items):
        item_path = os.path.join(DATASET_PATH, item)
        
        if os.path.isfile(item_path):
            if item.endswith(('.csv', '.xlsx', '.xls', '.txt', '.json')):
                data_files[item] = item_path
                
        elif os.path.isdir(item_path):
            # Check for data files in subdirectory
            subdir_files = [f for f in os.listdir(item_path) 
                           if f.endswith(('.csv', '.xlsx', '.xls', '.txt', '.json'))]
            if subdir_files:
                subdirs[item] = item_path
    
    print_subsection("FILES FOUND")
    
    if data_files:
        print(f"\n✓ Standalone data files: {len(data_files)}")
        for fname in data_files.keys():
            fsize = os.path.getsize(data_files[fname]) / (1024 * 1024)
            print(f"  - {fname} ({fsize:.2f} MB)")
    
    if subdirs:
        print(f"\n✓ Battery-specific subdirectories: {len(subdirs)}")
        for dirname in subdirs.keys():
            nfiles = len([f for f in os.listdir(subdirs[dirname]) 
                         if f.endswith(('.csv', '.xlsx', '.xls', '.txt', '.json'))])
            print(f"  - {dirname}/ ({nfiles} file(s))")
    
    if not data_files and not subdirs:
        raise ValueError("No data files found in dataset directory")
    
    # ========================================================================
    # LOAD STANDALONE FILES
    # ========================================================================
    
    dataframes = {}
    
    if data_files:
        print_subsection("LOADING STANDALONE FILES")
        
        for filename, filepath in data_files.items():
            print(f"\n{filename}...")
            try:
                df = _load_single_file(filepath)
                dataframes[filename] = df
                print(f"  ✓ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
                print(f"  Columns: {list(df.columns)}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    # ========================================================================
    # LOAD FILES FROM SUBDIRECTORIES
    # ========================================================================
    
    if subdirs:
        print_subsection("LOADING BATTERY-SPECIFIC SUBDIRECTORIES")
        
        for dirname, dirpath in subdirs.items():
            print(f"\n{dirname}/")
            
            battery_files = sorted([f for f in os.listdir(dirpath) 
                                   if f.endswith(('.csv', '.xlsx', '.xls', '.txt', '.json'))])
            
            for filename in battery_files:
                filepath = os.path.join(dirpath, filename)
                try:
                    df = _load_single_file(filepath)
                    key = f"{dirname}_{filename}"
                    dataframes[key] = df
                    print(f"  ✓ {filename}: {df.shape[0]} rows × {df.shape[1]} columns")
                    
                except Exception as e:
                    print(f"  ✗ {filename}: {e}")
    
    return dataframes

# ============================================================================
# LOAD SINGLE FILE
# ============================================================================

def _load_single_file(filepath):
    """
    Load a single data file in various formats.
    
    Parameters:
    -----------
    filepath : str
        Path to the file to load.
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe.
    
    Raises:
    -------
    ValueError
        If file format is not supported or loading fails.
    """
    
    if filepath.endswith('.csv'):
        # Try different delimiters
        try:
            df = pd.read_csv(filepath)
        except:
            try:
                df = pd.read_csv(filepath, sep=';')
            except:
                df = pd.read_csv(filepath, sep='\t')
    
    elif filepath.endswith(('.xlsx', '.xls')):
        # Try to load Excel file
        xls = pd.ExcelFile(filepath)
        # Use first sheet if multiple exist
        sheet_name = xls.sheet_names[0]
        df = pd.read_excel(filepath, sheet_name=sheet_name)
    
    elif filepath.endswith('.txt'):
        # Try different delimiters for text files
        try:
            df = pd.read_csv(filepath, sep='\s+')
        except:
            try:
                df = pd.read_csv(filepath, sep=',')
            except:
                df = pd.read_csv(filepath, sep='\t')
    
    elif filepath.endswith('.json'):
        df = pd.read_json(filepath)
    
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    return df

# ============================================================================
# ANALYZE LOADED DATA
# ============================================================================

def analyze_dataframes(dataframes):
    """
    Perform initial analysis of loaded dataframes.
    
    Parameters:
    -----------
    dataframes : dict
        Dictionary of loaded dataframes.
    
    Returns:
    --------
    pd.DataFrame
        Summary table of all dataframes.
    """
    
    print_subsection("SUMMARY OF LOADED DATA")
    
    summary_data = []
    
    for filename, df in dataframes.items():
        summary_data.append({
            'File': filename,
            'Rows': df.shape[0],
            'Columns': df.shape[1],
            'Column Names': ', '.join(df.columns),
            'Data Types': ', '.join(str(dt) for dt in df.dtypes),
            'Missing Values': df.isnull().sum().sum(),
            'Duplicated Rows': df.duplicated().sum()
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    if VERBOSE:
        print("\n")
        print(summary_df.to_string(index=False))
    
    return summary_df

# ============================================================================
# DETECT STRUCTURE
# ============================================================================

def detect_structure(dataframes):
    """
    Detect the structure of the loaded data.
    
    Parameters:
    -----------
    dataframes : dict
        Dictionary of loaded dataframes.
    
    Returns:
    --------
    dict
        Dictionary containing detected structure information.
    """
    
    print_subsection("DETECTED DATA STRUCTURE")
    
    structure = {
        'type': None,
        'n_files': len(dataframes),
        'dataframes': dataframes,
        'notes': []
    }
    
    # Check if single consolidated file or multiple battery files
    if len(dataframes) == 1:
        structure['type'] = 'single_consolidated'
        structure['notes'].append("Single consolidated data file detected")
        
        df = list(dataframes.values())[0]
        
        # Check for battery identifier column
        cols_lower = [c.lower() for c in df.columns]
        
        if any('battery' in c for c in cols_lower):
            structure['notes'].append("Battery identifier column present")
        if any('cycle' in c for c in cols_lower):
            structure['notes'].append("Cycle column present")
        if any('capacity' in c for c in cols_lower):
            structure['notes'].append("Capacity column present")
    
    else:
        structure['type'] = 'multiple_files'
        structure['notes'].append(f"Multiple files detected ({len(dataframes)} files)")
    
    if VERBOSE:
        print(f"\nType: {structure['type']}")
        for note in structure['notes']:
            print(f"  • {note}")
    
    return structure

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    try:
        # Load data
        dataframes = load_dataset()
        
        # Analyze loaded data
        summary = analyze_dataframes(dataframes)
        
        # Save summary
        summary_path = os.path.join(DATA_DIR, 'data_loading_summary.csv')
        summary.to_csv(summary_path, index=False)
        print(f"\n✓ Summary saved: {summary_path}")
        
        # Detect structure
        structure = detect_structure(dataframes)
        
        print_section("DATA LOADING COMPLETE")
        print(f"\n✓ Successfully loaded {len(dataframes)} dataframe(s)")
        print(f"✓ Output saved to: {DATA_DIR}")
        print("\nNext step: Run 02_data_processing.py")
        print("="*80 + "\n")
        
        return dataframes, structure
    
    except Exception as e:
        print(f"\n✗ Error during data loading: {e}")
        raise

if __name__ == "__main__":
    dataframes, structure = main()
