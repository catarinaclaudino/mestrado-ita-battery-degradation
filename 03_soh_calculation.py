"""
================================================================================
STEP 4: STATE OF HEALTH (SoH) CALCULATION
================================================================================

This script calculates capacity-based State of Health (SoH) for each battery.

Purpose:
- Calculate SoH using the first valid measured discharge capacity as reference
- SoH_i = (Q_i / Q_ref) × 100, where Q_ref = Q_1 (first measurement)
- Generate discharge capacity vs cycle plot
- Generate SoH vs cycle plot
- Create comprehensive summary tables
- Document all SoH calculations

IMPORTANT METHODOLOGICAL PRINCIPLES:
1. Q_ref is the FIRST VALID MEASURED discharge capacity for each battery
2. Do NOT normalize all batteries using a common population value
3. The first observation should correspond approximately to SoH = 100%
4. Preserve chronological order
5. Document reference capacity for each battery
6. Generate publication-quality figures and tables

================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (DATA_DIR, FIGURES_DIR, TABLES_DIR, VERBOSE, DIVIDER_WIDTH,
                    FIGURE_DPI, FIGURE_FORMAT, FONT_SIZE_TITLE, FONT_SIZE_LABEL, 
                    FONT_SIZE_TICK, LINE_WIDTH)

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
# LOAD PROCESSED DATA
# ============================================================================

def load_processed_data():
    """
    Load the processed data from Step 3.
    
    Returns:
    --------
    pd.DataFrame
        Processed dataframe with columns: battery_id, cycle, discharge_capacity
    """
    
    processed_path = os.path.join(DATA_DIR, 'processed_data.csv')
    
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed data not found: {processed_path}")
    
    df = pd.read_csv(processed_path)
    
    # Ensure correct data types
    df['battery_id'] = df['battery_id'].astype(str)
    df['cycle'] = df['cycle'].astype(int)
    df['discharge_capacity'] = pd.to_numeric(df['discharge_capacity'], errors='coerce')
    
    print(f"✓ Loaded processed data: {len(df)} observations")
    
    return df

# ============================================================================
# CALCULATE SoH
# ============================================================================

def calculate_soh(processed_df):
    """
    Calculate capacity-based State of Health for each battery.
    
    SoH_i = (Q_i / Q_ref) × 100
    
    where Q_ref = Q_1 (first valid measured discharge capacity)
    
    Parameters:
    -----------
    processed_df : pd.DataFrame
        Processed dataframe with columns: battery_id, cycle, discharge_capacity
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with SoH calculations: battery_id, cycle, discharge_capacity, SoH
    dict
        Reference capacity information for each battery
    """
    
    print_subsection("CALCULATING STATE OF HEALTH (SoH)")
    
    # Create a copy
    soh_df = processed_df.copy()
    
    # Initialize SoH column
    soh_df['SoH'] = np.nan
    
    # Reference capacities for each battery
    reference_capacities = {}
    
    # Process each battery independently
    unique_batteries = soh_df['battery_id'].unique()
    print(f"\nProcessing {len(unique_batteries)} battery/batteries:")
    
    for battery in sorted(unique_batteries):
        battery_mask = soh_df['battery_id'] == battery
        battery_data = soh_df[battery_mask].sort_values('cycle').reset_index(drop=True)
        
        # Get first valid (non-null) discharge capacity
        first_valid_idx = battery_data['discharge_capacity'].notna().idxmax()
        
        if pd.isna(battery_data.loc[first_valid_idx, 'discharge_capacity']):
            print(f"\n  {battery}: WARNING - No valid discharge capacity found")
            continue
        
        Q_ref = battery_data.loc[first_valid_idx, 'discharge_capacity']
        reference_capacities[battery] = Q_ref
        
        # Calculate SoH for all observations of this battery
        soh_values = (battery_data['discharge_capacity'] / Q_ref) * 100
        
        # Update SoH column in main dataframe
        soh_df.loc[battery_mask, 'SoH'] = soh_values.values
        
        # Print summary for this battery
        n_cycles = battery_data.shape[0]
        first_soh = soh_values.iloc[0]
        last_soh = soh_values.iloc[-1]
        capacity_loss = soh_values.iloc[0] - soh_values.iloc[-1]
        
        print(f"\n  {battery}:")
        print(f"    Reference capacity (Q_ref): {Q_ref:.6f}")
        print(f"    Number of valid cycles: {n_cycles}")
        print(f"    Initial SoH: {first_soh:.2f}%")
        print(f"    Final SoH: {last_soh:.2f}%")
        print(f"    Capacity loss: {capacity_loss:.2f}%")
        print(f"    Total capacity loss: {soh_values.iloc[0] - soh_values.iloc[-1]:.2f} percentage points")
    
    # Sort by battery and cycle
    soh_df.sort_values(['battery_id', 'cycle'], inplace=True)
    soh_df.reset_index(drop=True, inplace=True)
    
    return soh_df, reference_capacities

# ============================================================================
# VALIDATION OF SoH CALCULATION
# ============================================================================

def validate_soh(soh_df, reference_capacities):
    """
    Validate SoH calculations.
    
    Parameters:
    -----------
    soh_df : pd.DataFrame
        SoH dataframe
    reference_capacities : dict
        Reference capacities for each battery
    
    Returns:
    --------
    dict
        Validation results
    """
    
    print_subsection("VALIDATING SoH CALCULATIONS")
    
    validation = {
        'valid': True,
        'issues': []
    }
    
    # Check that first observation is approximately 100% SoH
    print("\nFirst observation SoH (should be ≈100%):")
    for battery in sorted(reference_capacities.keys()):
        battery_mask = soh_df['battery_id'] == battery
        first_soh = soh_df[battery_mask]['SoH'].iloc[0]
        print(f"  {battery}: {first_soh:.2f}%")
        
        if abs(first_soh - 100.0) > 0.1:
            validation['issues'].append(
                f"{battery}: First SoH is {first_soh:.2f}%, not ≈100%"
            )
    
    # Check that SoH is monotonically decreasing
    print("\nMonotonicity check:")
    for battery in sorted(reference_capacities.keys()):
        battery_mask = soh_df['battery_id'] == battery
        battery_data = soh_df[battery_mask].copy()
        
        # Calculate differences
        soh_diff = battery_data['SoH'].diff()
        
        # Count non-monotonic decreases
        increases = (soh_diff > 0).sum()
        
        if increases > 0:
            print(f"  {battery}: {increases} non-monotonic increases detected (warning)")
            validation['issues'].append(
                f"{battery}: {increases} SoH increases detected (expected monotonic decrease)"
            )
        else:
            print(f"  {battery}: Monotonic ✓")
    
    # Check for NaN values
    print("\nMissing values:")
    n_missing = soh_df['SoH'].isna().sum()
    if n_missing > 0:
        print(f"  Found {n_missing} missing SoH values")
        validation['issues'].append(f"Found {n_missing} missing SoH values")
    else:
        print(f"  None ✓")
    
    if validation['issues']:
        validation['valid'] = False
        print("\n⚠ Validation warnings:")
        for issue in validation['issues']:
            print(f"  - {issue}")
    else:
        print("\n✓ All validation checks passed")
    
    return validation

# ============================================================================
# GENERATE SUMMARY TABLES
# ============================================================================

def generate_soh_summary_table(soh_df, reference_capacities):
    """
    Generate comprehensive summary table for SoH calculations.
    
    Parameters:
    -----------
    soh_df : pd.DataFrame
        SoH dataframe
    reference_capacities : dict
        Reference capacities
    
    Returns:
    --------
    pd.DataFrame
        Summary table
    """
    
    print_subsection("SUMMARY TABLE: SoH CALCULATIONS")
    
    summary_rows = []
    
    for battery in sorted(reference_capacities.keys()):
        battery_mask = soh_df['battery_id'] == battery
        battery_data = soh_df[battery_mask]
        
        Q_ref = reference_capacities[battery]
        Q_initial = battery_data['discharge_capacity'].iloc[0]
        Q_final = battery_data['discharge_capacity'].iloc[-1]
        
        SoH_initial = battery_data['SoH'].iloc[0]
        SoH_final = battery_data['SoH'].iloc[-1]
        
        n_cycles = len(battery_data)
        
        summary_rows.append({
            'Battery ID': battery,
            'Reference Capacity (Ah)': f"{Q_ref:.6f}",
            'Initial Capacity (Ah)': f"{Q_initial:.6f}",
            'Final Capacity (Ah)': f"{Q_final:.6f}",
            'Capacity Loss (Ah)': f"{Q_initial - Q_final:.6f}",
            'Capacity Loss (%)': f"{((Q_initial - Q_final) / Q_ref) * 100:.2f}",
            'Initial SoH (%)': f"{SoH_initial:.2f}",
            'Final SoH (%)': f"{SoH_final:.2f}",
            'SoH Degradation (%)': f"{SoH_initial - SoH_final:.2f}",
            'Number of Cycles': n_cycles,
            'First Cycle': battery_data['cycle'].iloc[0],
            'Last Cycle': battery_data['cycle'].iloc[-1]
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    if VERBOSE:
        print("\n")
        print(summary_df.to_string(index=False))
    
    return summary_df

# ============================================================================
# GENERATE FIGURES
# ============================================================================

def generate_figures(soh_df, reference_capacities):
    """
    Generate publication-quality figures for SoH analysis.
    
    Parameters:
    -----------
    soh_df : pd.DataFrame
        SoH dataframe
    reference_capacities : dict
        Reference capacities
    """
    
    print_subsection("GENERATING FIGURES")
    
    # ========================================================================
    # Figure 1: Discharge Capacity vs Cycle
    # ========================================================================
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(reference_capacities)))
    
    for idx, battery in enumerate(sorted(reference_capacities.keys())):
        battery_mask = soh_df['battery_id'] == battery
        battery_data = soh_df[battery_mask]
        
        ax.plot(battery_data['cycle'], battery_data['discharge_capacity'],
                marker='o', linewidth=LINE_WIDTH, label=battery,
                color=colors[idx], markersize=4)
    
    ax.set_xlabel('Cycle Number', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Discharge Capacity (Ah)', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Raw Discharge Capacity Degradation', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_SIZE_LEGEND)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f'01_discharge_capacity_vs_cycle.{FIGURE_FORMAT}')
    plt.savefig(fig_path, dpi=FIGURE_DPI)
    print(f"✓ Figure 1 saved: 01_discharge_capacity_vs_cycle.{FIGURE_FORMAT}")
    plt.close()
    
    # ========================================================================
    # Figure 2: SoH vs Cycle
    # ========================================================================
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for idx, battery in enumerate(sorted(reference_capacities.keys())):
        battery_mask = soh_df['battery_id'] == battery
        battery_data = soh_df[battery_mask]
        
        ax.plot(battery_data['cycle'], battery_data['SoH'],
                marker='o', linewidth=LINE_WIDTH, label=battery,
                color=colors[idx], markersize=4)
    
    # Add 80% failure threshold line
    ax.axhline(y=80.0, color='red', linestyle='--', linewidth=2, label='80% SoH Failure Criterion')
    
    ax.set_xlabel('Cycle Number', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('State of Health (%)', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Capacity-Based State of Health (SoH) vs Cycle', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.set_ylim([75, 105])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_SIZE_LEGEND)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f'02_soh_vs_cycle.{FIGURE_FORMAT}')
    plt.savefig(fig_path, dpi=FIGURE_DPI)
    print(f"✓ Figure 2 saved: 02_soh_vs_cycle.{FIGURE_FORMAT}")
    plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(soh_df, summary_table):
    """
    Save SoH calculation results to files.
    
    Parameters:
    -----------
    soh_df : pd.DataFrame
        Complete SoH dataframe
    summary_table : pd.DataFrame
        Summary table
    """
    
    print_subsection("SAVING RESULTS")
    
    # Save SoH dataframe
    soh_path = os.path.join(DATA_DIR, 'soh_data.csv')
    soh_df.to_csv(soh_path, index=False)
    print(f"✓ SoH data saved: {soh_path}")
    
    # Save summary table
    summary_path = os.path.join(TABLES_DIR, 'Table_01_SoH_Summary.csv')
    summary_table.to_csv(summary_path, index=False)
    print(f"✓ Summary table saved: {summary_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print_section("STEP 4: STATE OF HEALTH (SoH) CALCULATION")
    
    try:
        # Load processed data
        print("\nLoading processed data...")
        processed_df = load_processed_data()
        
        # Calculate SoH
        soh_df, reference_capacities = calculate_soh(processed_df)
        
        # Validate calculations
        validation = validate_soh(soh_df, reference_capacities)
        
        # Generate summary table
        summary_table = generate_soh_summary_table(soh_df, reference_capacities)
        
        # Generate figures
        generate_figures(soh_df, reference_capacities)
        
        # Save results
        save_results(soh_df, summary_table)
        
        print_section("SoH CALCULATION COMPLETE")
        print(f"\n✓ Successfully calculated SoH for {len(reference_capacities)} battery/batteries")
        print(f"✓ Total observations: {len(soh_df)}")
        print(f"✓ Figures saved to: {FIGURES_DIR}")
        print(f"✓ Tables saved to: {TABLES_DIR}")
        print("\nNext step: Run 03_exploratory_analysis.py")
        print("="*DIVIDER_WIDTH + "\n")
        
        return soh_df, reference_capacities
    
    except Exception as e:
        print(f"\n✗ Error during SoH calculation: {e}")
        raise

if __name__ == "__main__":
    soh_df, reference_capacities = main()
