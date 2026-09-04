"""
================================================================================
STEP 7: RESIDUAL ANALYSIS AND MODEL VALIDATION
================================================================================

This script performs:
1. Residual diagnostics (residuals vs cycle, vs fitted values, histogram, Q-Q plot)
2. Temporal validation using chronological split (70% training, 30% validation)
3. Shapiro-Wilk test for residual normality
4. Assessment of residual behavior and model adequacy

IMPORTANT METHODOLOGICAL PRINCIPLES:
1. Shapiro-Wilk is used ONLY for assessing residual normality
2. Do NOT use Shapiro-Wilk for detecting Potential Failure
3. Validation observations are kept completely separate from fitting
4. Temporal validation preserves chronological order (no random shuffling)
5. Model is fitted ONLY on training data
6. Validation metrics are calculated on held-out test data

================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import stats
import pickle
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (DATA_DIR, FIGURES_DIR, TABLES_DIR, VERBOSE, DIVIDER_WIDTH,
                    TRAIN_FRACTION, ALPHA, FIGURE_DPI, FIGURE_FORMAT,
                    FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_TICK, LINE_WIDTH)

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
# TEMPORAL VALIDATION
# ============================================================================

def perform_temporal_validation(X, y, model, train_fraction=TRAIN_FRACTION):
    """
    Perform chronological temporal validation.
    
    Parameters:
    -----------
    X : array-like
        Cycle numbers (already sorted chronologically)
    y : array-like
        SoH values
    model : DegradationModel
        Fitted model
    train_fraction : float
        Fraction of data used for training (default 0.70)
    
    Returns:
    --------
    dict
        Validation results
    """
    
    print_subsection("TEMPORAL MODEL VALIDATION")
    
    # Split data chronologically
    n_total = len(X)
    n_train = int(np.floor(n_total * train_fraction))
    n_validation = n_total - n_train
    
    print(f"\nTotal observations: {n_total}")
    print(f"Training set (first {train_fraction*100:.0f}%): {n_train} observations")
    print(f"Validation set (remaining {(1-train_fraction)*100:.0f}%): {n_validation} observations")
    
    # Training and validation indices
    train_indices = np.arange(0, n_train)
    validation_indices = np.arange(n_train, n_total)
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[validation_indices]
    y_val = y[validation_indices]
    
    print(f"\nTraining cycle range: {X_train.min():.0f} - {X_train.max():.0f}")
    print(f"Validation cycle range: {X_val.min():.0f} - {X_val.max():.0f}")
    
    # Generate predictions in validation region
    y_val_pred = model.predict(X_val)
    
    # Calculate validation metrics
    val_residuals = y_val - y_val_pred
    val_rmse = np.sqrt(np.mean(val_residuals**2))
    val_mae = np.mean(np.abs(val_residuals))
    val_mean_error = np.mean(val_residuals)
    val_max_abs_error = np.max(np.abs(val_residuals))
    
    print(f"\nValidation Metrics:")
    print(f"  RMSE: {val_rmse:.6f}")
    print(f"  MAE: {val_mae:.6f}")
    print(f"  Mean Error: {val_mean_error:.6f}")
    print(f"  Max Absolute Error: {val_max_abs_error:.6f}")
    
    return {
        'train_indices': train_indices,
        'validation_indices': validation_indices,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'y_val_pred': y_val_pred,
        'val_residuals': val_residuals,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'val_mean_error': val_mean_error,
        'val_max_abs_error': val_max_abs_error
    }

# ============================================================================
# RESIDUAL DIAGNOSTICS
# ============================================================================

def generate_residual_diagnostics(model, X, y):
    """
    Generate residual diagnostic plots.
    
    Parameters:
    -----------
    model : DegradationModel
        Fitted model
    X : array-like
        Cycle numbers
    y : array-like
        SoH values
    """
    
    print_subsection("RESIDUAL DIAGNOSTIC PLOTS")
    
    residuals = model.residuals
    y_pred = model.predictions
    
    # Figure: Residuals vs Cycle
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, residuals, s=30, alpha=0.6, color='blue')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Cycle Number', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Residuals (%)', fontsize=FONT_SIZE_LABEL)
    ax.set_title(f'{model.name}: Residuals vs Cycle', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f'06_residuals_vs_cycle_{model.name.replace(" ", "_")}.{FIGURE_FORMAT}')
    plt.savefig(fig_path, dpi=FIGURE_DPI)
    print(f"✓ Residuals vs Cycle saved")
    plt.close()
    
    # Figure: Residuals vs Fitted Values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, s=30, alpha=0.6, color='blue')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Fitted SoH (%)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Residuals (%)', fontsize=FONT_SIZE_LABEL)
    ax.set_title(f'{model.name}: Residuals vs Fitted Values', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f'07_residuals_vs_fitted_{model.name.replace(" ", "_")}.{FIGURE_FORMAT}')
    plt.savefig(fig_path, dpi=FIGURE_DPI)
    print(f"✓ Residuals vs Fitted Values saved")
    plt.close()
    
    # Figure: Residual Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=20, color='blue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Residuals (%)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Frequency', fontsize=FONT_SIZE_LABEL)
    ax.set_title(f'{model.name}: Distribution of Residuals', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f'08_residual_histogram_{model.name.replace(" ", "_")}.{FIGURE_FORMAT}')
    plt.savefig(fig_path, dpi=FIGURE_DPI)
    print(f"✓ Residual Histogram saved")
    plt.close()
    
    # Figure: Q-Q Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(f'{model.name}: Normal Q-Q Plot', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f'09_qq_plot_{model.name.replace(" ", "_")}.{FIGURE_FORMAT}')
    plt.savefig(fig_path, dpi=FIGURE_DPI)
    print(f"✓ Q-Q Plot saved")
    plt.close()

# ============================================================================
# SHAPIRO-WILK TEST
# ============================================================================

def perform_shapiro_wilk_test(model, alpha=ALPHA):
    """
    Perform Shapiro-Wilk test for residual normality.
    
    Parameters:
    -----------
    model : DegradationModel
        Fitted model
    alpha : float
        Significance level (default 0.05)
    
    Returns:
    --------
    dict
        Shapiro-Wilk test results
    """
    
    print_subsection("SHAPIRO-WILK RESIDUAL NORMALITY TEST")
    
    residuals = model.residuals
    
    # Perform Shapiro-Wilk test
    W_statistic, p_value = stats.shapiro(residuals)
    
    # Decision
    if p_value < alpha:
        decision = "REJECT H0: Residuals are NOT normally distributed"
        is_normal = False
    else:
        decision = "FAIL TO REJECT H0: Residuals appear normally distributed"
        is_normal = True
    
    print(f"\nTest Configuration:")
    print(f"  Model: {model.name}")
    print(f"  H0: Residuals follow a normal distribution")
    print(f"  H1: Residuals do NOT follow a normal distribution")
    print(f"  Significance level (α): {alpha}")
    print(f"\nResults:")
    print(f"  W statistic: {W_statistic:.6f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Decision (α={alpha}): {decision}")
    
    # Additional statistics
    print(f"\nResidual Statistics:")
    print(f"  Mean: {np.mean(residuals):.6f}")
    print(f"  Std Dev: {np.std(residuals):.6f}")
    print(f"  Skewness: {stats.skew(residuals):.6f}")
    print(f"  Kurtosis: {stats.kurtosis(residuals):.6f}")
    
    return {
        'model_name': model.name,
        'W_statistic': W_statistic,
        'p_value': p_value,
        'alpha': alpha,
        'decision': decision,
        'is_normal': is_normal,
        'mean_residuals': np.mean(residuals),
        'std_residuals': np.std(residuals),
        'skewness': stats.skew(residuals),
        'kurtosis': stats.kurtosis(residuals)
    }

# ============================================================================
# VALIDATION FIGURE
# ============================================================================

def generate_validation_figure(X, y, model, validation_results):
    """
    Generate temporal validation figure.
    
    Parameters:
    -----------
    X : array-like
        Cycle numbers
    y : array-like
        SoH values
    model : DegradationModel
        Fitted model
    validation_results : dict
        Validation results
    """
    
    print_subsection("GENERATING VALIDATION FIGURE")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Training data and fit
    train_idx = validation_results['train_indices']
    val_idx = validation_results['validation_indices']
    
    ax.scatter(X[train_idx], y[train_idx], s=50, alpha=0.7, 
              label='Training Data', color='blue', marker='o')
    ax.scatter(X[val_idx], y[val_idx], s=50, alpha=0.7, 
              label='Validation Data', color='green', marker='s')
    
    # Fitted model on training region
    X_sorted = np.sort(X[train_idx])
    y_train_pred = model.predict(X_sorted)
    ax.plot(X_sorted, y_train_pred, 'b-', linewidth=LINE_WIDTH, label='Fitted Model (Training)')
    
    # Extrapolated model on validation region
    X_val_sorted = np.sort(X[val_idx])
    y_val_pred = validation_results['y_val_pred']
    # Sort validation predictions
    val_sort_idx = np.argsort(X[val_idx])
    ax.plot(X[val_idx][val_sort_idx], y_val_pred[val_sort_idx], 'g--', 
           linewidth=LINE_WIDTH, label='Extrapolated Model (Validation)')
    
    # 80% SoH threshold
    ax.axhline(y=80.0, color='red', linestyle='--', linewidth=2, label='80% SoH Failure Criterion')
    
    ax.set_xlabel('Cycle Number', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('State of Health (%)', fontsize=FONT_SIZE_LABEL)
    ax.set_title(f'{model.name}: Temporal Validation (Train: {TRAIN_FRACTION*100:.0f}%, Validation: {(1-TRAIN_FRACTION)*100:.0f}%)',
                fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_SIZE_LEGEND)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f'10_temporal_validation_{model.name.replace(" ", "_")}.{FIGURE_FORMAT}')
    plt.savefig(fig_path, dpi=FIGURE_DPI)
    print(f"✓ Temporal validation figure saved")
    plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_validation_results(shapiro_results, validation_results, model_name):
    """
    Save validation and diagnostics results.
    
    Parameters:
    -----------
    shapiro_results : dict
        Shapiro-Wilk test results
    validation_results : dict
        Temporal validation results
    model_name : str
        Name of the selected model
    """
    
    print_subsection("SAVING RESULTS")
    
    # Save Shapiro-Wilk results
    shapiro_df = pd.DataFrame([shapiro_results])
    shapiro_path = os.path.join(TABLES_DIR, 'Table_06_Shapiro_Wilk_Normality_Test.csv')
    shapiro_df.to_csv(shapiro_path, index=False)
    print(f"✓ Shapiro-Wilk test results saved")
    
    # Save validation metrics
    validation_df = pd.DataFrame([{
        'Model': model_name,
        'Train Observations': len(validation_results['X_train']),
        'Validation Observations': len(validation_results['X_val']),
        'Validation RMSE': f"{validation_results['val_rmse']:.6f}",
        'Validation MAE': f"{validation_results['val_mae']:.6f}",
        'Mean Validation Error': f"{validation_results['val_mean_error']:.6f}",
        'Max Absolute Error': f"{validation_results['val_max_abs_error']:.6f}"
    }])
    val_path = os.path.join(TABLES_DIR, 'Table_07_Temporal_Validation_Metrics.csv')
    validation_df.to_csv(val_path, index=False)
    print(f"✓ Temporal validation metrics saved")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print_section("STEP 7: RESIDUAL ANALYSIS AND TEMPORAL VALIDATION")
    
    try:
        # Load data and models
        print("\nLoading data and models...")
        soh_df, models, battery_info = load_data_and_models()
        
        # Get the best model (by R²)
        best_model_name = None
        best_r_squared = -np.inf
        for name, model in models.items():
            if not np.isnan(model.r_squared) and model.r_squared > best_r_squared:
                best_r_squared = model.r_squared
                best_model_name = name
        
        best_model = models[best_model_name]
        
        print(f"✓ Selected model for validation: {best_model_name} (R² = {best_r_squared:.6f})")
        
        # Prepare data
        X = battery_info['X']
        y = battery_info['y']
        
        # Perform temporal validation
        validation_results = perform_temporal_validation(X, y, best_model)
        
        # Generate residual diagnostics
        generate_residual_diagnostics(best_model, X, y)
        
        # Perform Shapiro-Wilk test
        shapiro_results = perform_shapiro_wilk_test(best_model)
        
        # Generate validation figure
        generate_validation_figure(X, y, best_model, validation_results)
        
        # Save results
        save_validation_results(shapiro_results, validation_results, best_model_name)
        
        print_section("RESIDUAL ANALYSIS AND VALIDATION COMPLETE")
        print(f"\n✓ Residual diagnostics generated")
        print(f"✓ Shapiro-Wilk p-value: {shapiro_results['p_value']:.6f}")
        print(f"✓ Residuals normal: {'Yes' if shapiro_results['is_normal'] else 'No'}")
        print(f"✓ Validation RMSE: {validation_results['val_rmse']:.6f}")
        print(f"✓ Validation MAE: {validation_results['val_mae']:.6f}")
        print(f"✓ Figures saved to: {FIGURES_DIR}")
        print(f"✓ Tables saved to: {TABLES_DIR}")
        print("\nNext step: Run 07_pettitt_change_point.py")
        print("="*DIVIDER_WIDTH + "\n")
        
        return best_model, validation_results, shapiro_results
    
    except Exception as e:
        print(f"\n✗ Error during residual analysis: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    best_model, validation_results, shapiro_results = main()
