"""
================================================================================
STEP 10: FAILURE ANALYSIS, RUL, AND P-F INTERVAL
================================================================================

This script:
1. Estimates Functional Failure cycle using 80% SoH criterion
2. Estimates Remaining Useful Life (RUL)
3. Identifies Potential Failure point if criteria are satisfied
4. Calculates P-F interval
5. Generates comprehensive failure analysis figures and tables

IMPORTANT DISTINCTIONS:
1. RUL = N_F_hat - N_c (remaining cycles from current point to failure)
2. P-F = N_F - N_P (interval between Potential Failure and Functional Failure)
3. These are DIFFERENT quantities answering DIFFERENT questions
4. 80% SoH is a CONVENTIONAL criterion for this research, not universal
5. Potential Failure requires BOTH statistical significance AND persistence
6. Functional Failure is interpolation or extrapolation as appropriate

================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import fsolve
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (DATA_DIR, FIGURES_DIR, TABLES_DIR, VERBOSE, DIVIDER_WIDTH,
                    SOH_FAILURE_THRESHOLD, FIGURE_DPI, FIGURE_FORMAT,
                    FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_TICK, LINE_WIDTH)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_section(title):
    if VERBOSE:
        print(f"\n{'='*DIVIDER_WIDTH}")
        print(title)
        print('='*DIVIDER_WIDTH)

def print_subsection(title):
    if VERBOSE:
        print(f"\n{'-'*DIVIDER_WIDTH}")
        print(title)
        print('-'*DIVIDER_WIDTH)

# ============================================================================
# LOAD DATA
# ============================================================================

def load_data_and_models():
    """Load SoH data and fitted models."""
    soh_path = os.path.join(DATA_DIR, 'soh_data.csv')
    models_path = os.path.join(DATA_DIR, 'fitted_models.pkl')
    battery_info_path = os.path.join(DATA_DIR, 'battery_data_info.pkl')
    
    soh_df = pd.read_csv(soh_path)
    with open(models_path, 'rb') as f:
        models = pickle.load(f)
    with open(battery_info_path, 'rb') as f:
        battery_info = pickle.load(f)
    
    return soh_df, models, battery_info

# ============================================================================
# ESTIMATE FAILURE CYCLE
# ============================================================================

def estimate_failure_cycle(model, soh_threshold=SOH_FAILURE_THRESHOLD):
    """
    Estimate the failure cycle at which SoH reaches the threshold.
    
    Parameters:
    -----------
    model : DegradationModel
        Fitted degradation model
    soh_threshold : float
        SoH threshold for failure (default 80%)
    
    Returns:
    --------
    dict
        Failure cycle estimation results
    """
    
    print_subsection("FAILURE CYCLE ESTIMATION")
    
    print(f"\nFunctional Failure Criterion (Adopted for this research):")
    print(f"  SoH_F = {soh_threshold}%")
    print(f"  Note: 80% is a CONVENTIONAL criterion, not a universal definition")
    
    # Try to find cycle where SoH = threshold
    # For simplicity, we'll search numerically
    
    # First check if threshold is within observed data
    min_soh = np.min(model.predictions)
    max_soh = np.max(model.predictions)
    
    print(f"\nObserved SoH Range:")
    print(f"  Max SoH: {max_soh:.2f}%")
    print(f"  Min SoH: {min_soh:.2f}%")
    
    if min_soh < soh_threshold <= max_soh:
        print(f"\n✓ Threshold {soh_threshold}% is within observed data (INTERPOLATION)")
        
        # Find cycle where SoH crosses threshold
        # by finding where model prediction equals threshold
        def objective(N):
            return model.predict(np.array([N]))[0] - soh_threshold
        
        try:
            # Initial guess
            x0 = 1000.0
            N_F = fsolve(objective, x0)[0]
            is_extrapolation = False
        except:
            N_F = np.nan
    elif max_soh >= soh_threshold > min_soh:
        print(f"✓ Threshold {soh_threshold}% requires EXTRAPOLATION")
        
        def objective(N):
            return model.predict(np.array([N]))[0] - soh_threshold
        
        try:
            x0 = 2000.0
            N_F = fsolve(objective, x0)[0]
            is_extrapolation = True
        except:
            N_F = np.nan
    else:
        print(f"✗ Threshold {soh_threshold}% is above maximum observed SoH")
        print(f"  Cannot estimate failure cycle reliably")
        N_F = np.nan
        is_extrapolation = None
    
    if not np.isnan(N_F):
        print(f"\nFailure Cycle Estimation Result:")
        print(f"  N_F_hat = {N_F:.2f} cycles")
        print(f"  Type: {'EXTRAPOLATION' if is_extrapolation else 'INTERPOLATION'}")
    else:
        print(f"\n✗ Could not estimate failure cycle")
    
    return {
        'N_F_hat': N_F,
        'is_extrapolation': is_extrapolation,
        'soh_threshold': soh_threshold,
        'min_observed_soh': min_soh,
        'max_observed_soh': max_soh
    }

# ============================================================================
# ESTIMATE RUL
# ============================================================================

def estimate_rul(N_F_hat, current_cycle):
    """
    Estimate Remaining Useful Life (RUL) from a current cycle.
    
    RUL(N_c) = N_F_hat - N_c
    
    Parameters:
    -----------
    N_F_hat : float
        Predicted failure cycle
    current_cycle : float
        Current cycle
    
    Returns:
    --------
    float
        RUL in cycles
    """
    
    if np.isnan(N_F_hat):
        return np.nan
    return max(0, N_F_hat - current_cycle)

# ============================================================================
# GENERATE FAILURE ANALYSIS FIGURE
# ============================================================================

def generate_failure_analysis_figure(X, y, model, failure_results, potential_failure_cycle=None):
    """
    Generate comprehensive failure analysis figure.
    
    Parameters:
    -----------
    X : array-like
        Cycle numbers
    y : array-like
        SoH values
    model : DegradationModel
        Fitted model
    failure_results : dict
        Failure cycle estimation results
    potential_failure_cycle : float, optional
        Potential Failure cycle (if identified)
    """
    
    print_subsection("GENERATING FAILURE ANALYSIS FIGURE")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot observed data
    ax.scatter(X, y, s=50, alpha=0.7, label='Observed Data', color='blue', marker='o')
    
    # Plot fitted model
    X_range = np.linspace(X.min(), X.max() * 1.5, 500)
    y_fit = model.predict(X_range)
    ax.plot(X_range, y_fit, 'b-', linewidth=LINE_WIDTH, label='Fitted Model')
    
    # Plot 80% SoH threshold
    ax.axhline(y=failure_results['soh_threshold'], color='red', linestyle='--', 
              linewidth=2, label=f"{failure_results['soh_threshold']}% SoH Failure Criterion")
    
    # Plot predicted failure cycle
    if not np.isnan(failure_results['N_F_hat']):
        ax.axvline(x=failure_results['N_F_hat'], color='red', linestyle=':', 
                  linewidth=2, alpha=0.7, label=f"Predicted Failure: N_F = {failure_results['N_F_hat']:.0f}")
    
    # Plot Potential Failure if it exists
    if potential_failure_cycle is not None and not np.isnan(potential_failure_cycle):
        ax.axvline(x=potential_failure_cycle, color='orange', linestyle='--', 
                  linewidth=2, alpha=0.7, label=f"Potential Failure: N_P = {potential_failure_cycle:.0f}")
        
        # Shade P-F interval
        if not np.isnan(failure_results['N_F_hat']):
            ax.axvspan(potential_failure_cycle, failure_results['N_F_hat'], 
                      alpha=0.2, color='orange', label='P-F Interval')
    
    # End of observed data
    ax.axvline(x=X.max(), color='gray', linestyle=':', linewidth=1.5, 
              alpha=0.7, label='End of Observed Data')
    
    ax.set_xlabel('Cycle Number', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('State of Health (%)', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Failure Analysis: Functional Failure and P-F Interval',
                fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc='lower left')
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.set_ylim([75, 105])
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f'13_failure_analysis.{FIGURE_FORMAT}')
    plt.savefig(fig_path, dpi=FIGURE_DPI)
    print(f"✓ Failure analysis figure saved")
    plt.close()

# ============================================================================
# GENERATE RUL FIGURE
# ============================================================================

def generate_rul_figure(X, y, model, failure_results):
    """
    Generate RUL vs cycle figure.
    
    Parameters:
    -----------
    X : array-like
        Cycle numbers
    y : array-like
        SoH values
    model : DegradationModel
        Fitted model
    failure_results : dict
        Failure cycle estimation results
    """
    
    print_subsection("GENERATING RUL FIGURE")
    
    if np.isnan(failure_results['N_F_hat']):
        print("⚠ Cannot generate RUL figure: failure cycle not estimated")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate RUL for each cycle
    X_range = np.linspace(X.min(), X.max(), len(X))
    RUL_values = failure_results['N_F_hat'] - X_range
    RUL_values = np.maximum(RUL_values, 0)  # RUL cannot be negative
    
    ax.plot(X_range, RUL_values, 'b-', linewidth=LINE_WIDTH, label='Predicted RUL')
    ax.scatter(X, failure_results['N_F_hat'] - X, s=50, alpha=0.7, 
              color='blue', marker='o', label='RUL at Observed Cycles')
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(x=failure_results['N_F_hat'], color='red', linestyle='--', 
              linewidth=2, alpha=0.7, label='Failure Cycle')
    
    ax.set_xlabel('Current Cycle Number', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Remaining Useful Life (RUL) [cycles]', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Remaining Useful Life (RUL) vs Current Cycle',
                fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_SIZE_LEGEND)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f'14_rul_vs_cycle.{FIGURE_FORMAT}')
    plt.savefig(fig_path, dpi=FIGURE_DPI)
    print(f"✓ RUL figure saved")
    plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_failure_results(failure_results, pf_interval, model_name, X):
    """
    Save failure analysis results.
    
    Parameters:
    -----------
    failure_results : dict
        Failure cycle estimation results
    pf_interval : float
        P-F interval (or None)
    model_name : str
        Model name
    X : array-like
        Cycle numbers
    """
    
    print_subsection("SAVING RESULTS")
    
    # Calculate RUL at end of data
    N_last = X.max()
    RUL_at_end = estimate_rul(failure_results['N_F_hat'], N_last)
    
    failure_df = pd.DataFrame([{
        'Model': model_name,
        'Predicted Failure Cycle (N_F_hat)': f"{failure_results['N_F_hat']:.2f}" if not np.isnan(failure_results['N_F_hat']) else "N/A",
        'Is Extrapolation': 'Yes' if failure_results['is_extrapolation'] else 'No' if failure_results['is_extrapolation'] is not None else 'N/A',
        'SoH Failure Threshold (%)': failure_results['soh_threshold'],
        'Min Observed SoH (%)': f"{failure_results['min_observed_soh']:.2f}",
        'Max Observed SoH (%)': f"{failure_results['max_observed_soh']:.2f}",
        'RUL at End of Observed Data': f"{RUL_at_end:.2f}" if not np.isnan(RUL_at_end) else "N/A",
        'P-F Interval': f"{pf_interval:.2f}" if pf_interval is not None and not np.isnan(pf_interval) else "N/A"
    }])
    
    failure_path = os.path.join(TABLES_DIR, 'Table_10_Failure_and_RUL_Analysis.csv')
    failure_df.to_csv(failure_path, index=False)
    print(f"✓ Failure analysis results saved")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print_section("STEP 10: FAILURE ANALYSIS, RUL, AND P-F INTERVAL")
    
    try:
        # Load data and models
        print("\nLoading data and models...")
        soh_df, models, battery_info = load_data_and_models()
        
        # Get the best model
        best_model_name = None
        best_r_squared = -np.inf
        for name, model in models.items():
            if not np.isnan(model.r_squared) and model.r_squared > best_r_squared:
                best_r_squared = model.r_squared
                best_model_name = name
        
        best_model = models[best_model_name]
        print(f"✓ Using model: {best_model_name}")
        
        # Prepare data
        X = battery_info['X']
        y = battery_info['y']
        
        # Estimate failure cycle
        failure_results = estimate_failure_cycle(best_model, soh_failure_threshold=SOH_FAILURE_THRESHOLD)
        
        # For now, P-F interval is not calculated (requires Potential Failure detection)
        # This will be done in the next step
        pf_interval = None
        potential_failure_cycle = None
        
        # Generate figures
        generate_failure_analysis_figure(X, y, best_model, failure_results, potential_failure_cycle)
        generate_rul_figure(X, y, best_model, failure_results)
        
        # Save results
        save_failure_results(failure_results, pf_interval, best_model_name, X)
        
        print_section("FAILURE ANALYSIS COMPLETE")
        print(f"\n✓ Failure cycle estimated: N_F_hat = {failure_results['N_F_hat']:.2f} cycles" if not np.isnan(failure_results['N_F_hat']) else "\n✗ Could not estimate failure cycle")
        print(f"✓ Figures saved to: {FIGURES_DIR}")
        print(f"✓ Tables saved to: {TABLES_DIR}")
        print("\nNext step: Run 10_final_summary.py")
        print("="*DIVIDER_WIDTH + "\n")
        
        return failure_results, pf_interval
    
    except Exception as e:
        print(f"\n✗ Error during failure analysis: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    failure_results, pf_interval = main()
