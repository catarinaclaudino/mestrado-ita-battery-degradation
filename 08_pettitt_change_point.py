"""
================================================================================
STEP 9: PETTITT CHANGE-POINT DETECTION
================================================================================

This script applies Pettitt's non-parametric change-point test to detect
Potential Failure in the residual sequence.

IMPORTANT METHODOLOGICAL PRINCIPLES:
1. Pettitt test is applied to RESIDUALS, not raw SoH trajectory
2. Change point represents shift in residual behavior
3. Potential Failure ≠ physical onset of electrochemical degradation
4. Potential Failure ≠ arbitrary SoH threshold
5. Statistical significance (p < 0.05) is necessary but NOT sufficient
6. Persistence criterion must ALSO be satisfied
7. If no significant persistent change is found, report it explicitly
8. Do NOT force detection of Potential Failure

================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import stats
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (DATA_DIR, FIGURES_DIR, TABLES_DIR, VERBOSE, DIVIDER_WIDTH,
                    ALPHA, PERSISTENCE_WINDOW, FIGURE_DPI, FIGURE_FORMAT,
                    FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_TICK, LINE_WIDTH,
                    MARKER_SIZE)

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
# PETTITT'S TEST IMPLEMENTATION
# ============================================================================

def pettitt_test(residuals, alpha=ALPHA):
    """
    Perform Pettitt's non-parametric change-point test.
    
    The test detects a single change point in a time series.
    
    Null Hypothesis (H0): No change point exists in the sequence
    Alternative Hypothesis (H1): A change point exists
    
    Parameters:
    -----------
    residuals : array-like
        Residual sequence
    alpha : float
        Significance level (default 0.05)
    
    Returns:
    --------
    dict
        Pettitt test results
    """
    
    print_subsection("PETTITT'S CHANGE-POINT TEST")
    
    n = len(residuals)
    print(f"\nTest Configuration:")
    print(f"  Sequence length: {n}")
    print(f"  Significance level (α): {alpha}")
    print(f"  H0: No change point in residual sequence")
    print(f"  H1: A change point exists in residual sequence")
    
    # Calculate U statistic for each potential change point
    U_values = np.zeros(n)
    
    for t in range(1, n):
        # For each position t, calculate U_t,n
        U_t = 0
        for i in range(1, t+1):
            for j in range(t+1, n+1):
                # Sign function
                if residuals[i-1] > residuals[j-1]:
                    U_t += 1
                elif residuals[i-1] < residuals[j-1]:
                    U_t -= 1
        
        U_values[t] = abs(U_t)
    
    # Find maximum U (the detected change point)
    U_max = np.max(U_values)
    t_star = np.argmax(U_values)
    
    print(f"\nChange-Point Detection:")
    print(f"  Maximum U statistic: {U_max:.0f}")
    print(f"  Detected change point: t* = {t_star}")
    print(f"  (between observation {t_star} and {t_star+1})")
    
    # Calculate approximate p-value using Pettitt's formula
    # p ≈ 2*exp(-6*U^2 / (n^3 + n^2))
    denominator = n**3 + n**2
    p_value = 2.0 * np.exp(-6.0 * U_max**2 / denominator)
    
    # Ensure p-value is in valid range
    p_value = min(p_value, 1.0)
    p_value = max(p_value, 0.0)
    
    print(f"\nStatistical Test Results:")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Decision (α={alpha}): ", end="")
    
    is_significant = p_value < alpha
    if is_significant:
        print(f"REJECT H0 - Change point is statistically significant")
    else:
        print(f"FAIL TO REJECT H0 - No significant change point detected")
    
    return {
        'U_max': U_max,
        't_star': t_star,
        'p_value': p_value,
        'alpha': alpha,
        'is_significant': is_significant,
        'U_values': U_values
    }

# ============================================================================
# PERSISTENCE CRITERION
# ============================================================================

def evaluate_persistence(residuals, t_star, persistence_window=PERSISTENCE_WINDOW):
    """
    Evaluate persistence criterion for detected change point.
    
    The change must remain distinguishable from the preceding regime
    over a predefined number of subsequent observations.
    
    Parameters:
    -----------
    residuals : array-like
        Residual sequence
    t_star : int
        Detected change point index
    persistence_window : int
        Number of observations after change point to evaluate (default 5)
    
    Returns:
    --------
    dict
        Persistence evaluation results
    """
    
    print_subsection("PERSISTENCE CRITERION EVALUATION")
    
    n = len(residuals)
    
    print(f"\nPersistence Criterion:")
    print(f"  The detected change point must be distinguishable from")
    print(f"  the preceding regime over {persistence_window} subsequent observations.")
    print(f"\nConfiguration:")
    print(f"  Detected change point: t* = {t_star}")
    print(f"  Persistence window: {persistence_window} observations")
    
    # Check if we have enough observations after change point
    if t_star + persistence_window >= n:
        available = n - t_star - 1
        print(f"\n⚠ WARNING: Only {available} observations after change point")
        print(f"           Persistence window requires {persistence_window}")
        print(f"           Cannot fully evaluate persistence criterion")
        persistence_satisfied = False
        explanation = f"Insufficient data: only {available} obs after change point"
    else:
        # Calculate mean residuals before and after change point
        residuals_before = residuals[:t_star]
        residuals_after = residuals[t_star:t_star+persistence_window]
        
        mean_before = np.mean(residuals_before)
        mean_after = np.mean(residuals_after)
        
        # Check if the sign or magnitude of residuals changed
        # Simple criterion: means should have different signs or
        # the post-change observations should be consistently on one side
        
        # Count how many post-change observations are on different side
        different_side = 0
        for res in residuals_after:
            if (mean_before >= 0 and res < 0) or (mean_before < 0 and res > 0):
                different_side += 1
        
        ratio_different = different_side / len(residuals_after)
        
        print(f"\nResidual Characteristics:")
        print(f"  Mean before change point: {mean_before:.6f}")
        print(f"  Mean after change point (window): {mean_after:.6f}")
        print(f"  Observations on different side: {different_side}/{len(residuals_after)}")
        print(f"  Ratio: {ratio_different:.2%}")
        
        # Persistence satisfied if there's clear distinction
        # (at least 60% of post-change observations on different side)
        persistence_satisfied = ratio_different >= 0.6
        
        if persistence_satisfied:
            print(f"\n✓ Persistence criterion SATISFIED")
            explanation = f"Change persists with {ratio_different:.1%} of observations on different side"
        else:
            print(f"\n✗ Persistence criterion NOT satisfied")
            explanation = f"Change not persistent enough ({ratio_different:.1%} on different side)"
    
    return {
        'persistence_satisfied': persistence_satisfied,
        'persistence_window': persistence_window,
        'explanation': explanation
    }

# ============================================================================
# LOAD DATA AND MODELS
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
# GENERATE PETTITT VISUALIZATION
# ============================================================================

def generate_pettitt_figure(residuals, t_star, pettitt_results, persistence_results):
    """
    Generate figure showing Pettitt test results on residuals.
    
    Parameters:
    -----------
    residuals : array-like
        Residual sequence
    t_star : int
        Detected change point
    pettitt_results : dict
        Pettitt test results
    persistence_results : dict
        Persistence evaluation results
    """
    
    print_subsection("GENERATING PETTITT VISUALIZATION")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Residuals with change point marked
    ax = axes[0]
    cycles = np.arange(len(residuals))
    ax.scatter(cycles, residuals, s=MARKER_SIZE, alpha=0.6, color='blue', label='Residuals')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=t_star, color='red', linestyle='--', linewidth=2, label=f'Detected Change Point (t*={t_star})')
    
    # Shade before and after regions
    ax.axvspan(0, t_star, alpha=0.1, color='blue', label='Before Change')
    ax.axvspan(t_star, len(residuals), alpha=0.1, color='red', label='After Change')
    
    ax.set_xlabel('Observation Index', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Residuals (%)', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Pettitt Change-Point Test: Residuals with Detected Change Point',
                fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_SIZE_LEGEND)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    
    # Plot 2: U statistic values
    ax = axes[1]
    U_values = pettitt_results['U_values']
    ax.plot(np.arange(len(U_values)), U_values, 'b-', linewidth=LINE_WIDTH, label='|U_t,n|')
    ax.scatter([t_star], [U_values[t_star]], s=100, color='red', marker='*', 
              label=f'Maximum U = {pettitt_results["U_max"]:.0f} at t*={t_star}', zorder=5)
    
    ax.set_xlabel('Potential Change Point Position (t)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('|U Statistic|', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Pettitt U Statistic Across All Possible Change Points',
                fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_SIZE_LEGEND)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    
    # Add text box with results
    textstr = f"""Pettitt Test Results:
p-value = {pettitt_results['p_value']:.6f}
Significant (α=0.05): {'Yes' if pettitt_results['is_significant'] else 'No'}

Persistence Criterion:
{persistence_results['explanation']}
Satisfied: {'Yes' if persistence_results['persistence_satisfied'] else 'No'}"""
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f'12_pettitt_change_point_detection.{FIGURE_FORMAT}')
    plt.savefig(fig_path, dpi=FIGURE_DPI)
    print(f"✓ Pettitt visualization saved")
    plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_pettitt_results(pettitt_results, persistence_results, model_name):
    """
    Save Pettitt change-point detection results.
    
    Parameters:
    -----------
    pettitt_results : dict
        Pettitt test results
    persistence_results : dict
        Persistence evaluation results
    model_name : str
        Model name
    """
    
    print_subsection("SAVING RESULTS")
    
    pettitt_df = pd.DataFrame([{
        'Model': model_name,
        'Detected Change Point (t*)': pettitt_results['t_star'],
        'U Statistic': f"{pettitt_results['U_max']:.0f}",
        'p-value': f"{pettitt_results['p_value']:.6f}",
        'Significant (α=0.05)': 'Yes' if pettitt_results['is_significant'] else 'No',
        'Persistence Window': persistence_results['persistence_window'],
        'Persistence Satisfied': 'Yes' if persistence_results['persistence_satisfied'] else 'No',
        'Explanation': persistence_results['explanation']
    }])
    
    pettitt_path = os.path.join(TABLES_DIR, 'Table_09_Pettitt_Change_Point_Detection.csv')
    pettitt_df.to_csv(pettitt_path, index=False)
    print(f"✓ Pettitt test results saved")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print_section("STEP 9: PETTITT CHANGE-POINT DETECTION")
    
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
        
        # Get residuals
        residuals = best_model.residuals
        print(f"✓ Residual sequence length: {len(residuals)}")
        
        # Perform Pettitt test
        pettitt_results = pettitt_test(residuals, alpha=ALPHA)
        
        # Evaluate persistence
        persistence_results = evaluate_persistence(residuals, pettitt_results['t_star'], 
                                                   persistence_window=PERSISTENCE_WINDOW)
        
        # Generate visualization
        generate_pettitt_figure(residuals, pettitt_results['t_star'], 
                               pettitt_results, persistence_results)
        
        # Save results
        save_pettitt_results(pettitt_results, persistence_results, best_model_name)
        
        print_section("PETTITT CHANGE-POINT DETECTION COMPLETE")
        print(f"\n✓ Change point detected at t* = {pettitt_results['t_star']}")
        print(f"✓ p-value: {pettitt_results['p_value']:.6f}")
        print(f"✓ Significant: {'Yes' if pettitt_results['is_significant'] else 'No'}")
        print(f"✓ Persistence criterion: {'Satisfied' if persistence_results['persistence_satisfied'] else 'NOT satisfied'}")
        print(f"✓ Figures saved to: {FIGURES_DIR}")
        print(f"✓ Tables saved to: {TABLES_DIR}")
        print("\nNext step: Run 09_rul_and_pf_interval.py")
        print("="*DIVIDER_WIDTH + "\n")
        
        return pettitt_results, persistence_results
    
    except Exception as e:
        print(f"\n✗ Error during Pettitt analysis: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    pettitt_results, persistence_results = main()
