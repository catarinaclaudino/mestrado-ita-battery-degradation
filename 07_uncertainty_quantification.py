"""
================================================================================
STEP 8: UNCERTAINTY QUANTIFICATION
================================================================================

This script quantifies model prediction uncertainty.

Methods:
1. Calculate residual standard deviation
2. Construct prediction intervals around fitted model
3. Distinguish confidence intervals from prediction intervals
4. Evaluate how uncertainty grows with extrapolation
5. Generate uncertainty visualization

IMPORTANT METHODOLOGICAL PRINCIPLES:
1. Clearly distinguish model uncertainty from measurement variability
2. Confidence intervals are for model parameters
3. Prediction intervals account for both model and residual uncertainty
4. Uncertainty increases with distance from observed data (extrapolation)
5. Document assumptions about error distribution

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
                    TRAIN_FRACTION, FIGURE_DPI, FIGURE_FORMAT,
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
# CALCULATE UNCERTAINTY METRICS
# ============================================================================

def calculate_uncertainty_metrics(model, X, train_fraction=TRAIN_FRACTION):
    """
    Calculate residual standard deviation and prediction uncertainty.
    
    Parameters:
    -----------
    model : DegradationModel
        Fitted model
    X : array-like
        Cycle numbers
    train_fraction : float
        Fraction used for training
    
    Returns:
    --------
    dict
        Uncertainty metrics
    """
    
    print_subsection("UNCERTAINTY QUANTIFICATION")
    
    residuals = model.residuals
    n = len(residuals)
    p = model.n_params
    
    # Residual standard deviation (standard error of regression)
    s_e = np.sqrt(np.sum(residuals**2) / (n - p - 1))
    
    print(f"\nResidual Standard Deviation (s_e):")
    print(f"  s_e = sqrt[Σ(e_i)² / (n - p - 1)]")
    print(f"  s_e = sqrt[{np.sum(residuals**2):.6f} / ({n} - {p} - 1)]")
    print(f"  s_e = {s_e:.6f}")
    
    # Degrees of freedom
    df = n - p - 1
    
    print(f"\nDegrees of Freedom:")
    print(f"  n = {n} (total observations)")
    print(f"  p = {p} (model parameters)")
    print(f"  df = n - p - 1 = {df}")
    
    # Calculate mean cycle
    X_mean = np.mean(X)
    
    # Calculate sum of squared deviations
    S_xx = np.sum((X - X_mean)**2)
    
    print(f"\nData Statistics:")
    print(f"  Mean cycle (X_mean): {X_mean:.2f}")
    print(f"  Sum of squared deviations (S_xx): {S_xx:.2f}")
    
    return {
        's_e': s_e,
        'n': n,
        'p': p,
        'df': df,
        'X_mean': X_mean,
        'S_xx': S_xx
    }

# ============================================================================
# CALCULATE PREDICTION INTERVALS
# ============================================================================

def calculate_prediction_intervals(model, X, X_pred, uncertainty_metrics, confidence=0.95):
    """
    Calculate prediction intervals for future predictions.
    
    For linear model: PI = y_pred ± t_alpha * s_e * sqrt(1 + 1/n + (X-X_mean)²/S_xx)
    
    Parameters:
    -----------
    model : DegradationModel
        Fitted model
    X : array-like
        Training cycle numbers
    X_pred : array-like
        Prediction cycle numbers
    uncertainty_metrics : dict
        Uncertainty metrics
    confidence : float
        Confidence level (default 0.95 for 95% PI)
    
    Returns:
    --------
    dict
        Prediction intervals
    """
    
    print_subsection("CALCULATING PREDICTION INTERVALS")
    
    s_e = uncertainty_metrics['s_e']
    n = uncertainty_metrics['n']
    p = uncertainty_metrics['p']
    df = uncertainty_metrics['df']
    X_mean = uncertainty_metrics['X_mean']
    S_xx = uncertainty_metrics['S_xx']
    
    # Critical t-value for two-sided interval
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha/2, df)
    
    print(f"\nPrediction Interval Configuration:")
    print(f"  Confidence level: {confidence*100:.0f}%")
    print(f"  Alpha: {alpha}")
    print(f"  Critical t-value (t_{df}): {t_crit:.4f}")
    
    # Get predictions
    y_pred = model.predict(X_pred)
    
    # Calculate standard error of prediction for each X value
    se_pred = s_e * np.sqrt(1 + 1/n + (X_pred - X_mean)**2 / S_xx)
    
    # Prediction interval
    margin = t_crit * se_pred
    pi_lower = y_pred - margin
    pi_upper = y_pred + margin
    
    print(f"\nPrediction Interval Calculation:")
    print(f"  PI = y_pred ± t_crit × s_e × sqrt(1 + 1/n + (X-X_mean)²/S_xx)")
    print(f"\nExample (at mean cycle):")
    mean_idx = np.argmin(np.abs(X_pred - X_mean))
    print(f"  At X = {X_pred[mean_idx]:.0f}:")
    print(f"    y_pred = {y_pred[mean_idx]:.4f}")
    print(f"    SE = {se_pred[mean_idx]:.6f}")
    print(f"    Margin = {margin[mean_idx]:.6f}")
    print(f"    PI: [{pi_lower[mean_idx]:.4f}, {pi_upper[mean_idx]:.4f}]")
    
    return {
        'y_pred': y_pred,
        'pi_lower': pi_lower,
        'pi_upper': pi_upper,
        'se_pred': se_pred,
        'margin': margin,
        't_crit': t_crit
    }

# ============================================================================
# GENERATE UNCERTAINTY FIGURE
# ============================================================================

def generate_uncertainty_figure(X, y, model, X_pred, pi_results, uncertainty_metrics):
    """
    Generate figure showing model fit with prediction intervals.
    
    Parameters:
    -----------
    X : array-like
        Training cycle numbers
    y : array-like
        Training SoH values
    model : DegradationModel
        Fitted model
    X_pred : array-like
        Prediction cycle numbers
    pi_results : dict
        Prediction interval results
    uncertainty_metrics : dict
        Uncertainty metrics
    """
    
    print_subsection("GENERATING UNCERTAINTY FIGURE")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot observed data
    ax.scatter(X, y, s=50, alpha=0.7, label='Observed Data', color='blue', marker='o')
    
    # Plot fitted model
    X_sorted = np.sort(X)
    y_fit = model.predict(X_sorted)
    ax.plot(X_sorted, y_fit, 'b-', linewidth=LINE_WIDTH, label='Fitted Model')
    
    # Plot prediction intervals
    X_pred_sorted = np.sort(X_pred)
    pi_lower_sorted = pi_results['pi_lower'][np.argsort(X_pred)]
    pi_upper_sorted = pi_results['pi_upper'][np.argsort(X_pred)]
    y_pred_sorted = pi_results['y_pred'][np.argsort(X_pred)]
    
    ax.fill_between(X_pred_sorted, pi_lower_sorted, pi_upper_sorted, 
                    alpha=0.3, color='green', label='95% Prediction Interval')
    ax.plot(X_pred_sorted, y_pred_sorted, 'g--', linewidth=LINE_WIDTH, alpha=0.7, label='Predicted Mean')
    
    # Add 80% SoH threshold
    ax.axhline(y=80.0, color='red', linestyle='--', linewidth=2, label='80% SoH Failure Criterion')
    
    # Add separation line between observed and extrapolation
    ax.axvline(x=X.max(), color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='End of Observed Data')
    
    ax.set_xlabel('Cycle Number', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('State of Health (%)', fontsize=FONT_SIZE_LABEL)
    ax.set_title(f'{model.name}: Prediction Intervals (95% Confidence)',
                fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc='lower left')
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.set_ylim([75, 105])
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f'11_prediction_intervals_{model.name.replace(" ", "_")}.{FIGURE_FORMAT}')
    plt.savefig(fig_path, dpi=FIGURE_DPI)
    print(f"✓ Uncertainty figure saved")
    plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_uncertainty_results(uncertainty_metrics, model_name):
    """
    Save uncertainty quantification results.
    
    Parameters:
    -----------
    uncertainty_metrics : dict
        Uncertainty metrics
    model_name : str
        Model name
    """
    
    print_subsection("SAVING RESULTS")
    
    uncertainty_df = pd.DataFrame([{
        'Model': model_name,
        'Residual Std Dev (s_e)': f"{uncertainty_metrics['s_e']:.6f}",
        'N Observations': uncertainty_metrics['n'],
        'N Parameters': uncertainty_metrics['p'],
        'Degrees of Freedom': uncertainty_metrics['df'],
        'Mean Cycle': f"{uncertainty_metrics['X_mean']:.2f}",
        'Sum Sq Deviations': f"{uncertainty_metrics['S_xx']:.2f}"
    }])
    
    unc_path = os.path.join(TABLES_DIR, 'Table_08_Uncertainty_Metrics.csv')
    uncertainty_df.to_csv(unc_path, index=False)
    print(f"✓ Uncertainty metrics saved")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print_section("STEP 8: UNCERTAINTY QUANTIFICATION")
    
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
        
        # Generate extended prediction range (100% beyond observed data)
        X_max = X.max()
        X_pred = np.linspace(X.min(), X_max * 1.5, 200)
        
        # Calculate uncertainty metrics
        uncertainty_metrics = calculate_uncertainty_metrics(best_model, X)
        
        # Calculate prediction intervals
        pi_results = calculate_prediction_intervals(best_model, X, X_pred, uncertainty_metrics)
        
        # Generate uncertainty figure
        generate_uncertainty_figure(X, y, best_model, X_pred, pi_results, uncertainty_metrics)
        
        # Save results
        save_uncertainty_results(uncertainty_metrics, best_model_name)
        
        print_section("UNCERTAINTY QUANTIFICATION COMPLETE")
        print(f"\n✓ Residual standard deviation (s_e): {uncertainty_metrics['s_e']:.6f}")
        print(f"✓ Prediction intervals calculated")
        print(f"✓ Figure saved to: {FIGURES_DIR}")
        print(f"✓ Tables saved to: {TABLES_DIR}")
        print("\nNext step: Run 08_failure_analysis.py")
        print("="*DIVIDER_WIDTH + "\n")
        
        return best_model, uncertainty_metrics
    
    except Exception as e:
        print(f"\n✗ Error during uncertainty quantification: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    best_model, uncertainty_metrics = main()
