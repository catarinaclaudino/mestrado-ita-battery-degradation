"""
================================================================================
STEP 3: DATA PROCESSING
================================================================================

This script processes raw CALCE CS2-35 data to extract valid aging cycles,
discharge capacity measurements, and cycle numbers.

Purpose:
- Identify discharge capacity measurements for each cycle
- Associate capacity with corresponding cycle number
- Preserve chronological order
- Identify missing, incomplete, or duplicated cycles
- Document all preprocessing decisions
- Create a processed dataset: cycle, discharge_capacity, battery_id

IMPORTANT METHODOLOGICAL PRINCIPLES:
1. Do NOT arbitrarily remove observations
2. Do NOT automatically remove outliers
3. Do NOT smooth the degradation trajectory
4. Preserve chronological nature of aging data
5. Document every preprocessing decision
6. Do NOT assume column names before inspecting actual data
7. If multiple discharge-capacity measurements exist for a cycle,
   determine which one represents the relevant capacity measurement

================================================================================
"""

import os
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, OUTPUT_BASE_DIR, VERBOSE, DIVIDER_WIDTH
from 01_load_data import load_dataset, detect_structure

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_section(title):
    """Print a formatted section header."""
    if VERBOSE:
        print(f"\n{'='*DIVIDER_WIDTH}")
        print(title)
        print('='*DIVIDER_WIDTH)

def print_subsection(title):
    """Print a formatted subsection header."""
    if VERBOSE:
        print(f"\n{'-'*DIVIDER_WIDTH}")
        print(title)
        print('-'*DIVIDER_WIDTH)

# ============================================================================
# COLUMN IDENTIFICATION
# ============================================================================

def identify_columns(df):
    """
    Identify relevant columns in the dataframe without assuming column names.
    
    Searches for columns containing:
    - Battery identifier
    - Cycle number
    - Discharge capacity
    - Other relevant measurements
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe to analyze
    
    Returns:
    --------
    dict
        Dictionary with column roles and actual column names
    """
    
    print_subsection("COLUMN IDENTIFICATION")
    
    column_mapping = {
        'battery_id': None,
        'cycle': None,
        'discharge_capacity': None,
        'voltage': None,
        'current': None,
        'temperature': None,
        'timestamp': None,
        'other': []
    }
    
    cols_lower = {col: col for col in df.columns}  # Preserve original names
    cols_lower_map = {col.lower(): col for col in df.columns}
    
    print(f"\nAvailable columns: {list(df.columns)}")
    print(f"\nData types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    print(f"\nFirst 5 rows:")
    print(df.head(5).to_string())
    
    # Search for battery identifier
    battery_keywords = ['battery', 'bat', 'cell', 'id', 'identifier']
    for keyword in battery_keywords:
        for col_lower, col_orig in cols_lower_map.items():
            if keyword in col_lower and column_mapping['battery_id'] is None:
                column_mapping['battery_id'] = col_orig
                print(f"\n✓ Battery ID column: {col_orig}")
                break
    
    # Search for cycle number
    cycle_keywords = ['cycle', 'cyc', 'n', 'test_num', 'step', 'loop']
    for keyword in cycle_keywords:
        for col_lower, col_orig in cols_lower_map.items():
            if keyword in col_lower and column_mapping['cycle'] is None:
                column_mapping['cycle'] = col_orig
                print(f"✓ Cycle column: {col_orig}")
                break
    
    # Search for discharge capacity
    capacity_keywords = ['capacity', 'discharge', 'discharge_cap', 'q', 'charge_capacity']
    for keyword in capacity_keywords:
        for col_lower, col_orig in cols_lower_map.items():
            if keyword in col_lower and column_mapping['discharge_capacity'] is None:
                column_mapping['discharge_capacity'] = col_orig
                print(f"✓ Discharge capacity column: {col_orig}")
                break
    
    # Search for other measurements
    voltage_keywords = ['voltage', 'volt', 'v']
    for keyword in voltage_keywords:
        for col_lower, col_orig in cols_lower_map.items():
            if keyword in col_lower and column_mapping['voltage'] is None:
                column_mapping['voltage'] = col_orig
                print(f"✓ Voltage column: {col_orig}")
                break
    
    current_keywords = ['current', 'curr', 'i']
    for keyword in current_keywords:
        for col_lower, col_orig in cols_lower_map.items():
            if keyword in col_lower and column_mapping['current'] is None:
                column_mapping['current'] = col_orig
                print(f"✓ Current column: {col_orig}")
                break
    
    temperature_keywords = ['temperature', 'temp', 't']
    for keyword in temperature_keywords:
        for col_lower, col_orig in cols_lower_map.items():
            if keyword in col_lower and column_mapping['temperature'] is None:
                column_mapping['temperature'] = col_orig
                print(f"✓ Temperature column: {col_orig}")
                break
    
    timestamp_keywords = ['time', 'timestamp', 'date', 'datetime']
    for keyword in timestamp_keywords:
        for col_lower, col_orig in cols_lower_map.items():
            if keyword in col_lower and column_mapping['timestamp'] is None:
                column_mapping['timestamp'] = col_orig
                print(f"✓ Timestamp column: {col_orig}")
                break
    
    # Record unmapped columns
    mapped_cols = set(v for v in column_mapping.values() if v is not None)
    for col in df.columns:
        if col not in mapped_cols:
            column_mapping['other'].append(col)
    
    if column_mapping['other']:
        print(f"\nUnmapped columns: {column_mapping['other']}")
    
    return column_mapping

# ============================================================================
# PROCESS DATA FOR SINGLE CONSOLIDATED FILE
# ============================================================================

def process_consolidated_data(df, columns_info):
    """
    Process data from a single consolidated dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe
    columns_info : dict
        Column mapping information
    
    Returns:
    --------
    pd.DataFrame
        Processed dataframe with columns: battery_id, cycle, discharge_capacity
    """
    
    print_subsection("PROCESSING CONSOLIDATED DATA")
    
    # Validate required columns
    required_columns = ['cycle', 'discharge_capacity']
    for col_type in required_columns:
        if columns_info[col_type] is None:
            raise ValueError(f"Could not identify {col_type} column. " +
                           f"Available columns: {list(df.columns)}")
    
    # Start with a copy
    processed = df.copy()
    
    # Rename columns to standard names
    rename_dict = {}
    if columns_info['cycle'] is not None:
        rename_dict[columns_info['cycle']] = 'cycle'
    if columns_info['discharge_capacity'] is not None:
        rename_dict[columns_info['discharge_capacity']] = 'discharge_capacity'
    if columns_info['battery_id'] is not None:
        rename_dict[columns_info['battery_id']] = 'battery_id'
    
    processed.rename(columns=rename_dict, inplace=True)
    
    # If no battery_id column, add one
    if 'battery_id' not in processed.columns:
        processed['battery_id'] = 'CS2-35'
        print("✓ Added default battery_id: CS2-35")
    
    # Sort by battery_id and cycle to preserve chronological order
    processed.sort_values(['battery_id', 'cycle'], inplace=True)
    processed.reset_index(drop=True, inplace=True)
    
    print(f"✓ Data sorted by battery_id and cycle")
    print(f"✓ Total observations: {len(processed)}")
    
    return processed[['battery_id', 'cycle', 'discharge_capacity']]

# ============================================================================
# IDENTIFY QUALITY ISSUES
# ============================================================================

def identify_quality_issues(processed_df):
    """
    Identify missing cycles, duplicates, and invalid observations.
    
    Parameters:
    -----------
    processed_df : pd.DataFrame
        Processed dataframe
    
    Returns:
    --------
    dict
        Dictionary containing quality issue information
    """
    
    print_subsection("DATA QUALITY ASSESSMENT")
    
    issues = {
        'missing_values': {},
        'duplicate_cycles': {},
        'invalid_observations': {},
        'cycle_gaps': {}
    }
    
    # Check for missing values
    print("\nMissing values:")
    for col in processed_df.columns:
        n_missing = processed_df[col].isnull().sum()
        if n_missing > 0:
            issues['missing_values'][col] = n_missing
            print(f"  {col}: {n_missing} missing")
        else:
            print(f"  {col}: 0 missing ✓")
    
    # Check for duplicated cycles per battery
    print("\nDuplicated cycles per battery:")
    for battery in processed_df['battery_id'].unique():
        battery_data = processed_df[processed_df['battery_id'] == battery]
        duplicated = battery_data['cycle'].duplicated().sum()
        if duplicated > 0:
            issues['duplicate_cycles'][battery] = duplicated
            print(f"  {battery}: {duplicated} duplicated cycles")
            # Show which cycles are duplicated
            dup_cycles = battery_data[battery_data['cycle'].duplicated(keep=False)]['cycle'].unique()
            print(f"    Duplicated cycle numbers: {sorted(dup_cycles)}")
        else:
            print(f"  {battery}: No duplicates ✓")
    
    # Check for invalid observations (negative or zero capacity)
    print("\nInvalid observations (negative/zero capacity):")
    invalid = processed_df[processed_df['discharge_capacity'] <= 0]
    if len(invalid) > 0:
        issues['invalid_observations']['zero_or_negative_capacity'] = len(invalid)
        print(f"  Found {len(invalid)} observations with capacity ≤ 0")
    else:
        print(f"  None found ✓")
    
    # Check for cycle gaps per battery
    print("\nCycle gaps per battery:")
    for battery in processed_df['battery_id'].unique():
        battery_data = processed_df[processed_df['battery_id'] == battery].sort_values('cycle')
        cycles = battery_data['cycle'].values
        
        # Check for gaps
        if len(cycles) > 1:
            cycle_diffs = np.diff(cycles)
            gaps = np.where(cycle_diffs > 1)[0]
            
            if len(gaps) > 0:
                issues['cycle_gaps'][battery] = len(gaps)
                print(f"  {battery}: {len(gaps)} gap(s)")
                for gap_idx in gaps[:5]:  # Show first 5 gaps
                    print(f"    Gap between cycle {cycles[gap_idx]} and {cycles[gap_idx+1]}")
                if len(gaps) > 5:
                    print(f"    ... and {len(gaps)-5} more gaps")
            else:
                print(f"  {battery}: No gaps ✓")
    
    return issues

# ============================================================================
# GENERATE PROCESSING SUMMARY
# ============================================================================

def generate_processing_summary(raw_df, processed_df, column_mapping, quality_issues):
    """
    Generate a summary table of the data processing step.
    
    Parameters:
    -----------
    raw_df : pd.DataFrame
        Raw dataframe
    processed_df : pd.DataFrame
        Processed dataframe
    column_mapping : dict
        Column identification results
    quality_issues : dict
        Quality issues identified
    
    Returns:
    --------
    pd.DataFrame
        Summary table
    """
    
    print_subsection("DATA PROCESSING SUMMARY TABLE")
    
    summary_data = {
        'Metric': [
            'Raw observations',
            'Processed observations',
            'Raw columns',
            'Processed columns',
            'Unique batteries',
            'Unique cycles (per battery)',
            'Missing values',
            'Duplicate cycles',
            'Invalid observations',
            'Cycle gaps'
        ],
        'Value': [
            len(raw_df),
            len(processed_df),
            raw_df.shape[1],
            processed_df.shape[1],
            processed_df['battery_id'].nunique(),
            processed_df.groupby('battery_id')['cycle'].nunique().mean(),
            sum(quality_issues['missing_values'].values()),
            sum(quality_issues['duplicate_cycles'].values()),
            sum(quality_issues['invalid_observations'].values()),
            sum(quality_issues['cycle_gaps'].values())
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    if VERBOSE:
        print("\n")
        print(summary_df.to_string(index=False))
    
    return summary_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print_section("STEP 3: DATA PROCESSING - CALCE CS2-35")
    
    try:
        # Load data
        print("\nLoading dataset...")
        dataframes, structure = load_dataset()
        
        # Get the main dataframe
        if len(dataframes) == 1:
            raw_df = list(dataframes.values())[0]
            print(f"✓ Using consolidated file: {list(dataframes.keys())[0]}")
        else:
            # If multiple files, use the largest one as primary
            sizes = {k: len(v) for k, v in dataframes.items()}
            primary_file = max(sizes, key=sizes.get)
            raw_df = dataframes[primary_file]
            print(f"✓ Using primary file: {primary_file} ({sizes[primary_file]} rows)")
        
        # Identify columns
        column_mapping = identify_columns(raw_df)
        
        # Process data
        print_subsection("PROCESSING DATA")
        processed_df = process_consolidated_data(raw_df, column_mapping)
        
        # Assess data quality
        quality_issues = identify_quality_issues(processed_df)
        
        # Generate summary
        summary_table = generate_processing_summary(raw_df, processed_df, 
                                                     column_mapping, quality_issues)
        
        # Save processed data
        processed_path = os.path.join(DATA_DIR, 'processed_data.csv')
        processed_df.to_csv(processed_path, index=False)
        print(f"\n✓ Processed data saved: {processed_path}")
        
        # Save summary
        summary_path = os.path.join(DATA_DIR, 'processing_summary.csv')
        summary_table.to_csv(summary_path, index=False)
        print(f"✓ Summary table saved: {summary_path}")
        
        print_section("DATA PROCESSING COMPLETE")
        print(f"\n✓ Successfully processed {len(processed_df)} observations")
        print(f"✓ Batteries: {processed_df['battery_id'].nunique()}")
        print(f"✓ Output saved to: {DATA_DIR}")
        print("\nNext step: Run 02_soh_calculation.py")
        print("="*DIVIDER_WIDTH + "\n")
        
        return processed_df, column_mapping, quality_issues
    
    except Exception as e:
        print(f"\n✗ Error during data processing: {e}")
        raise

if __name__ == "__main__":
    processed_df, column_mapping, quality_issues = main()
