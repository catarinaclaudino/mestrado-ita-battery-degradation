"""
================================================================================
STEP 5: EXPLORATORY DEGRADATION ANALYSIS
================================================================================

This script performs exploratory analysis of battery degradation before
fitting statistical models.

Purpose:
- Understand overall degradation trend
- Assess monotonicity of degradation
- Identify nonlinear behavior
- Detect changes in degradation rate
- Identify anomalies and outliers
- Identify regions of sparse data
- Evaluate extrapolation challenges
- Generate descriptive statistics

IMPORTANT:
Do NOT use exploratory transformations as the final degradation model
unless they are explicitly selected as candidate models.

================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (DATA_DIR, FIGURES_DIR, TABLES_DIR, VERBOSE, DIVIDER_WIDTH,
                    FIGURE_DPI, FIGURE_FORMAT, FONT_SIZE_TITLE, FONT_SIZE_LABEL,
                    FONT_SIZE_TICK, LINE_WIDTH, MARKER_SIZE)

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
# LOAD SoH DATA
# ============================================================================

def load_soh_data():
    """Load SoH data from Step 4."""
    soh_path = os.path.join(DATA_DIR, 'soh_data.csv')
    if not os.path.exists(soh_path):
        raise FileNotFoundError(f"SoH data not found: {soh_path}")
    return pd.read_csv(soh_path)

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================

def calculate_descriptive_statistics(soh_df):
    """
    Calculate descriptive statistics for SoH degradation.
    
    Parameters:
    -----------
    soh_df : pd.DataFrame
        SoH dataframe
    
    Returns:
    --------
    pd.DataFrame
        Descriptive statistics table
    """
    
    print_subsection("DESCRIPTIVE STATISTICS")
    
    stats_rows = []
    
    for battery in sorted(soh_df['battery_id'].unique()):
        battery_mask = soh_df['battery_id'] == battery
        battery_soh = soh_df[battery_mask]['SoH']
        battery_cap = soh_df[battery_mask]['discharge_capacity']
        battery_cycles = soh_df[battery_mask]['cycle']
        
        # Calculate statistics
        stats_rows.append({
            'Battery': battery,
            'Mean SoH (%)': f"{battery_soh.mean():.2f}",
            'Std SoH (%)': f"{battery_soh.std():.2f}",
            'Min SoH (%)': f"{battery_soh.min():.2f}",
            'Max SoH (%)': f"{battery_soh.max():.2f}",
            'Median SoH (%)': f"{battery_soh.median():.2f}",
            'Mean Capacity (Ah)': f"{battery_cap.mean():.6f}",
            'Std Capacity (Ah)': f"{battery_cap.std():.6f}",
            'Min Capacity (Ah)': f"{battery_cap.min():.6f}",
            'Max Capacity (Ah)': f"{battery_cap.max():.6f}",
            'Cycle Range': f"{battery_cycles.min()}-{battery_cycles.max()}",
            'N Observations': len(battery_soh)
        })
    
    stats_df = pd.DataFrame(stats_rows)
    
    if VERBOSE:
        print("\n")
        print(stats_df.to_string(index=False))
    
    return stats_df

# ============================================================================
# DEGRADATION RATE ANALYSIS
# ============================================================================

def analyze_degradation_rate(soh_df):
    """
    Analyze cycle-to-cycle degradation rates.
    
    Parameters:
    -----------
    soh_df : pd.DataFrame
        SoH dataframe
    
    Returns:
    --------
    pd.DataFrame
        Degradation rate statistics
    """
    
    print_subsection("DEGRADATION RATE ANALYSIS")
    
    rate_rows = []
    
    for battery in sorted(soh_df['battery_id'].unique()):
        battery_mask = soh_df['battery_id'] == battery
        battery_data = soh_df[battery_mask].sort_values('cycle')
        
        # Calculate cycle-to-cycle degradation
        soh_diff = battery_data['SoH'].diff()
        
        # Remove first NaN
        soh_diff = soh_diff[1:]
        
        rate_rows.append({
            'Battery': battery,
            'Mean Degradation Rate (%/cycle)': f"{soh_diff.mean():.6f}",
            'Std Degradation Rate (%/cycle)': f"{soh_diff.std():.6f}",
            'Min Degradation Rate (%/cycle)': f"{soh_diff.min():.6f}",
            'Max Degradation Rate (%/cycle)': f"{soh_diff.max():.6f}",
            'Median Degradation Rate (%/cycle)': f"{soh_diff.median():.6f}"
        })
    
    rate_df = pd.DataFrame(rate_rows)
    
    if VERBOSE:
        print("\n")
        print(rate_df.to_string(index=False))
    
    return rate_df

# ============================================================================
# MONOTONICITY ASSESSMENT
# ============================================================================

def assess_monotonicity(soh_df):
    """
    Assess monotonicity of degradation curves.
    
    Parameters:
    -----------
    soh_df : pd.DataFrame
        SoH dataframe
    
    Returns:
    --------
    pd.DataFrame
        Monotonicity assessment table
    """
    
    print_subsection("MONOTONICITY ASSESSMENT")
    
    mono_rows = []
    
    for battery in sorted(soh_df['battery_id'].unique()):
        battery_mask = soh_df['battery_id'] == battery
        battery_data = soh_df[battery_mask].sort_values('cycle')
        
        soh_values = battery_data['SoH'].values
        soh_diff = np.diff(soh_values)
        
        # Count increases and decreases
        n_decreases = (soh_diff < 0).sum()
        n_increases = (soh_diff > 0).sum()
        n_stable = (soh_diff == 0).sum()
        n_transitions = len(soh_diff)
        
        # Calculate monotonicity ratio
        if n_transitions > 0:
            monotonicity_ratio = n_decreases / n_transitions
        else:
            monotonicity_ratio = 0
        
        is_monotonic = n_increases == 0
        
        mono_rows.append({
            'Battery': battery,
            'Monotonic (Expected)': 'Yes' if is_monotonic else 'No',
            'N Transitions': n_transitions,
            'N Decreases': n_decreases,
            'N Increases': n_increases,
            'N Stable': n_stable,
            'Monotonicity Ratio': f"{monotonicity_ratio:.4f}"
        })
    
    mono_df = pd.DataFrame(mono_rows)
    
    if VERBOSE:
        print("\n")
        print(mono_df.to_string(index=False))
    
    return mono_df

# ============================================================================
# GENERATE EXPLORATORY FIGURES
# ============================================================================

def generate_exploratory_figures(soh_df):
    """
    Generate exploratory analysis figures.
    
    Parameters:
    -----------
    soh_df : pd.DataFrame
        SoH dataframe
    """
    
    print_subsection("GENERATING EXPLORATORY FIGURES")
    
    unique_batteries = sorted(soh_df['battery_id'].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_batteries)))
    
    # ========================================================================
    # Figure 3: Cycle-to-Cycle Degradation Rate
    # ========================================================================
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for idx, battery in enumerate(unique_batteries):
        battery_mask = soh_df['battery_id'] == battery
        battery_data = soh_df[battery_mask].sort_values('cycle')
        
        soh_diff = battery_data['SoH'].diff()
        cycles_diff = battery_data['cycle'].iloc[1:].values
        
        ax.scatter(cycles_diff, soh_diff.iloc[1:], 
                  label=battery, color=colors[idx], s=MARKER_SIZE, alpha=0.6)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Cycle Number', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Cycle-to-Cycle SoH Change (%/cycle)', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Cycle-to-Cycle Degradation Rate Variation', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f'03_degradation_rate_variation.{FIGURE_FORMAT}')
    plt.savefig(fig_path, dpi=FIGURE_DPI)
    print(f"✓ Figure 3 saved: 03_degradation_rate_variation.{FIGURE_FORMAT}")
    plt.close()
    
    # ========================================================================
    # Figure 4: Histogram of SoH Values
    # ========================================================================
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for idx, battery in enumerate(unique_batteries):
        battery_mask = soh_df['battery_id'] == battery
        battery_soh = soh_df[battery_mask]['SoH']
        
        ax.hist(battery_soh, bins=20, alpha=0.6, label=battery, color=colors[idx])
    
    ax.set_xlabel('State of Health (%)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Frequency', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Distribution of SoH Values', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f'04_soh_distribution.{FIGURE_FORMAT}')
    plt.savefig(fig_path, dpi=FIGURE_DPI)
    print(f"✓ Figure 4 saved: 04_soh_distribution.{FIGURE_FORMAT}")
    plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_exploratory_results(descriptive_stats, degradation_rates, monotonicity):
    """
    Save exploratory analysis results.
    
    Parameters:
    -----------
    descriptive_stats : pd.DataFrame
        Descriptive statistics
    degradation_rates : pd.DataFrame
        Degradation rate statistics
    monotonicity : pd.DataFrame
        Monotonicity assessment
    """
    
    print_subsection("SAVING RESULTS")
    
    # Save descriptive statistics
    stats_path = os.path.join(TABLES_DIR, 'Table_02_Descriptive_Statistics.csv')
    descriptive_stats.to_csv(stats_path, index=False)
    print(f"✓ Descriptive statistics saved: {stats_path}")
    
    # Save degradation rates
    rates_path = os.path.join(TABLES_DIR, 'Table_03_Degradation_Rates.csv')
    degradation_rates.to_csv(rates_path, index=False)
    print(f"✓ Degradation rates saved: {rates_path}")
    
    # Save monotonicity assessment
    mono_path = os.path.join(TABLES_DIR, 'Table_04_Monotonicity_Assessment.csv')
    monotonicity.to_csv(mono_path, index=False)
    print(f"✓ Monotonicity assessment saved: {mono_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print_section("STEP 5: EXPLORATORY DEGRADATION ANALYSIS")
    
    try:
        # Load SoH data
        print("\nLoading SoH data...")
        soh_df = load_soh_data()
        print(f"✓ Loaded {len(soh_df)} observations")
        
        # Calculate descriptive statistics
        descriptive_stats = calculate_descriptive_statistics(soh_df)
        
        # Analyze degradation rates
        degradation_rates = analyze_degradation_rate(soh_df)
        
        # Assess monotonicity
        monotonicity = assess_monotonicity(soh_df)
        
        # Generate figures
        generate_exploratory_figures(soh_df)
        
        # Save results
        save_exploratory_results(descriptive_stats, degradation_rates, monotonicity)
        
        print_section("EXPLORATORY ANALYSIS COMPLETE")
        print(f"\n✓ Analysis successful")
        print(f"✓ Figures saved to: {FIGURES_DIR}")
        print(f"✓ Tables saved to: {TABLES_DIR}")
        print("\nNext step: Run 05_degradation_models.py")
        print("="*DIVIDER_WIDTH + "\n")
        
        return soh_df
    
    except Exception as e:
        print(f"\n✗ Error during exploratory analysis: {e}")
        raise

if __name__ == "__main__":
    soh_df = main()
