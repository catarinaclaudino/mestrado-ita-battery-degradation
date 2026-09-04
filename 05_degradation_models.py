"""
================================================================================
STEP 6: STATISTICAL DEGRADATION MODELING
================================================================================

This script fits candidate degradation models to battery SoH data.

Candidate models:
- Linear
- Polynomial (degrees 2, 3, 4)
- Exponential
- Logarithmic
- Power-law

For each model, calculate:
- R²
- Adjusted R²
- RMSE
- MAE
- Residuals
- Number of parameters

Assess:
- Residual behavior
- Model parsimony
- Physical plausibility
- Monotonicity
- Extrapolation behavior

IMPORTANT:
Do NOT select model solely based on R².
Consider statistical fit + residual behavior + predictive performance + 
parsimony + physical plausibility.

================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import curve_fit
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (DATA_DIR, FIGURES_DIR, TABLES_DIR, VERBOSE, DIVIDER_WIDTH,
                    POLYNOMIAL_DEGREES, FIGURE_DPI, FIGURE_FORMAT,
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
# MODEL DEFINITIONS
# ============================================================================

class DegradationModel:
    """
    Base class for degradation models.
    """
    
    def __init__(self, name, formula):
        self.name = name
        self.formula = formula
        self.params = None
        self.predictions = None
        self.residuals = None
        self.r_squared = None
        self.adjusted_r_squared = None
        self.rmse = None
        self.mae = None
        self.n_params = 0
    
    def fit(self, X, y):
        """Fit model to data. Must be implemented by subclass."""
        raise NotImplementedError
    
    def predict(self, X):
        """Generate predictions. Must be implemented by subclass."""
        raise NotImplementedError
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate model performance metrics.
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        """
        
        # Residuals
        residuals = y_true - y_pred
        self.residuals = residuals
        
        # R²
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        self.r_squared = 1 - (ss_res / ss_tot)
        
        # Adjusted R²
        n = len(y_true)
        adj_r2 = 1 - ((1 - self.r_squared) * (n - 1) / (n - self.n_params - 1))
        self.adjusted_r_squared = adj_r2
        
        # RMSE
        self.rmse = np.sqrt(np.mean(residuals**2))
        
        # MAE
        self.mae = np.mean(np.abs(residuals))

# ============================================================================
# LINEAR MODEL
# ============================================================================

class LinearModel(DegradationModel):
    """
    Linear degradation model: SoH(N) = β0 + β1*N
    """
    
    def __init__(self):
        super().__init__('Linear', 'SoH(N) = β₀ + β₁·N')
        self.n_params = 2
    
    def fit(self, X, y):
        """Fit linear model using numpy polyfit."""
        # Add intercept column
        X_with_intercept = np.column_stack([np.ones_like(X), X])
        # Fit using least squares
        self.params = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        self.predictions = self.predict(X)
        self.calculate_metrics(y, self.predictions)
    
    def predict(self, X):
        """Generate predictions."""
        return self.params[0] + self.params[1] * X

# ============================================================================
# POLYNOMIAL MODELS
# ============================================================================

class PolynomialModel(DegradationModel):
    """
    Polynomial degradation model: SoH(N) = β0 + β1*N + β2*N² + ...
    """
    
    def __init__(self, degree):
        self.degree = degree
        super().__init__(f'Polynomial (degree {degree})',
                        f'SoH(N) = β₀ + Σ βᵢ·Nⁱ (i=1..{degree})')
        self.n_params = degree + 1
    
    def fit(self, X, y):
        """Fit polynomial model."""
        self.params = np.polyfit(X, y, self.degree)
        self.predictions = self.predict(X)
        self.calculate_metrics(y, self.predictions)
    
    def predict(self, X):
        """Generate predictions."""
        return np.polyval(self.params, X)

# ============================================================================
# EXPONENTIAL MODEL
# ============================================================================

class ExponentialModel(DegradationModel):
    """
    Exponential degradation model: SoH(N) = β0 * exp(-β1*N)
    """
    
    def __init__(self):
        super().__init__('Exponential', 'SoH(N) = β₀·exp(-β₁·N)')
        self.n_params = 2
    
    def fit(self, X, y):
        """Fit exponential model."""
        try:
            # Initial guess
            p0 = [100.0, 0.001]
            self.params, _ = curve_fit(self._func, X, y, p0=p0, maxfev=5000)
            self.predictions = self.predict(X)
            self.calculate_metrics(y, self.predictions)
        except:
            self.params = None
            self.predictions = None
            self.r_squared = np.nan
    
    def _func(self, N, b0, b1):
        """Exponential function."""
        return b0 * np.exp(-b1 * N)
    
    def predict(self, X):
        """Generate predictions."""
        if self.params is None:
            return np.full_like(X, np.nan)
        return self._func(X, *self.params)

# ============================================================================
# LOGARITHMIC MODEL
# ============================================================================

class LogarithmicModel(DegradationModel):
    """
    Logarithmic degradation model: SoH(N) = β0 - β1*ln(N)
    """
    
    def __init__(self):
        super().__init__('Logarithmic', 'SoH(N) = β₀ - β₁·ln(N)')
        self.n_params = 2
    
    def fit(self, X, y):
        """Fit logarithmic model."""
        try:
            X_log = np.log(X)
            X_with_intercept = np.column_stack([np.ones_like(X_log), X_log])
            self.params = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            self.predictions = self.predict(X)
            self.calculate_metrics(y, self.predictions)
        except:
            self.params = None
            self.predictions = None
            self.r_squared = np.nan
    
    def predict(self, X):
        """Generate predictions."""
        if self.params is None:
            return np.full_like(X, np.nan)
        return self.params[0] - self.params[1] * np.log(X)

# ============================================================================
# POWER-LAW MODEL
# ============================================================================

class PowerLawModel(DegradationModel):
    """
    Power-law degradation model: SoH(N) = β0 - β1*N^β2
    """
    
    def __init__(self):
        super().__init__('Power-Law', 'SoH(N) = β₀ - β₁·Nᵝ²')
        self.n_params = 3
    
    def fit(self, X, y):
        """Fit power-law model."""
        try:
            p0 = [100.0, 0.1, 0.5]
            self.params, _ = curve_fit(self._func, X, y, p0=p0, maxfev=5000)
            self.predictions = self.predict(X)
            self.calculate_metrics(y, self.predictions)
        except:
            self.params = None
            self.predictions = None
            self.r_squared = np.nan
    
    def _func(self, N, b0, b1, b2):
        """Power-law function."""
        return b0 - b1 * np.power(N, b2)
    
    def predict(self, X):
        """Generate predictions."""
        if self.params is None:
            return np.full_like(X, np.nan)
        return self._func(X, *self.params)

# ============================================================================
# LOAD DATA
# ============================================================================

def load_soh_data():
    """Load SoH data."""
    soh_path = os.path.join(DATA_DIR, 'soh_data.csv')
    if not os.path.exists(soh_path):
        raise FileNotFoundError(f"SoH data not found: {soh_path}")
    return pd.read_csv(soh_path)

# ============================================================================
# FIT CANDIDATE MODELS
# ============================================================================

def fit_candidate_models(X, y):
    """
    Fit all candidate degradation models.
    
    Parameters:
    -----------
    X : array-like
        Cycle numbers (independent variable)
    y : array-like
        SoH values (dependent variable)
    
    Returns:
    --------
    dict
        Dictionary of fitted models
    """
    
    print_subsection("FITTING CANDIDATE MODELS")
    
    models = {}
    
    # Linear model
    print("\nFitting Linear model...")
    linear = LinearModel()
    linear.fit(X, y)
    models['Linear'] = linear
    print(f"  ✓ R² = {linear.r_squared:.6f}")
    
    # Polynomial models
    for degree in POLYNOMIAL_DEGREES:
        print(f"\nFitting Polynomial model (degree {degree})...")
        poly = PolynomialModel(degree)
        poly.fit(X, y)
        models[f'Polynomial_{degree}'] = poly
        print(f"  ✓ R² = {poly.r_squared:.6f}")
    
    # Exponential model
    print("\nFitting Exponential model...")
    exp = ExponentialModel()
    exp.fit(X, y)
    models['Exponential'] = exp
    if exp.r_squared is not np.nan:
        print(f"  ✓ R² = {exp.r_squared:.6f}")
    else:
        print(f"  ✗ Failed to fit")
    
    # Logarithmic model
    print("\nFitting Logarithmic model...")
    log = LogarithmicModel()
    log.fit(X, y)
    models['Logarithmic'] = log
    if log.r_squared is not np.nan:
        print(f"  ✓ R² = {log.r_squared:.6f}")
    else:
        print(f"  ✗ Failed to fit")
    
    # Power-law model
    print("\nFitting Power-Law model...")
    powerlaw = PowerLawModel()
    powerlaw.fit(X, y)
    models['Power-Law'] = powerlaw
    if powerlaw.r_squared is not np.nan:
        print(f"  ✓ R² = {powerlaw.r_squared:.6f}")
    else:
        print(f"  ✗ Failed to fit")
    
    return models

# ============================================================================
# MODEL COMPARISON TABLE
# ============================================================================

def generate_model_comparison_table(models):
    """
    Generate comparison table for all fitted models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of fitted models
    
    Returns:
    --------
    pd.DataFrame
        Model comparison table
    """
    
    print_subsection("MODEL COMPARISON TABLE")
    
    comparison_rows = []
    
    for model_name, model in models.items():
        comparison_rows.append({
            'Model': model_name,
            'Formula': model.formula,
            'N Parameters': model.n_params,
            'R²': f"{model.r_squared:.6f}" if model.r_squared is not np.nan else "N/A",
            'Adjusted R²': f"{model.adjusted_r_squared:.6f}" if model.adjusted_r_squared is not np.nan else "N/A",
            'RMSE': f"{model.rmse:.6f}" if model.rmse is not np.nan else "N/A",
            'MAE': f"{model.mae:.6f}" if model.mae is not np.nan else "N/A"
        })
    
    comparison_df = pd.DataFrame(comparison_rows)
    
    if VERBOSE:
        print("\n")
        print(comparison_df.to_string(index=False))
    
    return comparison_df

# ============================================================================
# GENERATE FIGURE
# ============================================================================

def generate_model_comparison_figure(X, y, models):
    """
    Generate figure comparing all fitted models.
    
    Parameters:
    -----------
    X : array-like
        Cycle numbers
    y : array-like
        SoH values
    models : dict
        Dictionary of fitted models
    """
    
    print_subsection("GENERATING COMPARISON FIGURE")
    
    n_models = len(models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for idx, (model_name, model) in enumerate(models.items()):
        ax = axes[idx]
        
        # Plot data
        ax.scatter(X, y, s=30, alpha=0.6, label='Data', color='blue')
        
        # Plot model fit
        if model.predictions is not None and not np.all(np.isnan(model.predictions)):
            sorted_indices = np.argsort(X)
            ax.plot(X[sorted_indices], model.predictions[sorted_indices],
                   'r-', linewidth=LINE_WIDTH, label='Fitted Model')
            
            r2_text = f"R² = {model.r_squared:.6f}"
        else:
            r2_text = f"R² = N/A (fit failed)"
        
        ax.set_xlabel('Cycle Number', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('SoH (%)', fontsize=FONT_SIZE_LABEL)
        ax.set_title(f"{model_name}\n{r2_text}", fontsize=FONT_SIZE_LABEL, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=FONT_SIZE_TICK)
        ax.tick_params(labelsize=FONT_SIZE_TICK)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f'05_candidate_models_comparison.{FIGURE_FORMAT}')
    plt.savefig(fig_path, dpi=FIGURE_DPI)
    print(f"✓ Figure saved: 05_candidate_models_comparison.{FIGURE_FORMAT}")
    plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_models_and_results(models, comparison_table, battery_data):
    """
    Save fitted models and results.
    
    Parameters:
    -----------
    models : dict
        Dictionary of fitted models
    comparison_table : pd.DataFrame
        Model comparison table
    battery_data : dict
        Battery data information
    """
    
    print_subsection("SAVING RESULTS")
    
    # Save comparison table
    comparison_path = os.path.join(TABLES_DIR, 'Table_05_Model_Comparison.csv')
    comparison_table.to_csv(comparison_path, index=False)
    print(f"✓ Model comparison table saved: {comparison_path}")
    
    # Save models as pickle for later use
    import pickle
    models_path = os.path.join(DATA_DIR, 'fitted_models.pkl')
    with open(models_path, 'wb') as f:
        pickle.dump(models, f)
    print(f"✓ Fitted models saved: {models_path}")
    
    # Save battery data information
    battery_info_path = os.path.join(DATA_DIR, 'battery_data_info.pkl')
    with open(battery_info_path, 'wb') as f:
        pickle.dump(battery_data, f)
    print(f"✓ Battery data info saved: {battery_info_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print_section("STEP 6: STATISTICAL DEGRADATION MODELING")
    
    try:
        # Load SoH data
        print("\nLoading SoH data...")
        soh_df = load_soh_data()
        
        # For now, use the first battery as test case
        battery = soh_df['battery_id'].iloc[0]
        battery_mask = soh_df['battery_id'] == battery
        battery_data = soh_df[battery_mask].sort_values('cycle')
        
        print(f"✓ Using battery: {battery}")
        print(f"✓ Observations: {len(battery_data)}")
        
        # Prepare data
        X = battery_data['cycle'].values.astype(float)
        y = battery_data['SoH'].values.astype(float)
        
        # Fit candidate models
        models = fit_candidate_models(X, y)
        
        # Generate comparison table
        comparison_table = generate_model_comparison_table(models)
        
        # Generate comparison figure
        generate_model_comparison_figure(X, y, models)
        
        # Save results
        battery_info = {
            'battery_id': battery,
            'n_observations': len(battery_data),
            'X': X,
            'y': y
        }
        save_models_and_results(models, comparison_table, battery_info)
        
        print_section("DEGRADATION MODELING COMPLETE")
        print(f"\n✓ Successfully fitted {len(models)} candidate models")
        print(f"✓ Best model by R²: {comparison_table.loc[comparison_table['R²'].notna()].sort_values('R²', ascending=False).iloc[0]['Model']}")
        print(f"✓ Figures saved to: {FIGURES_DIR}")
        print(f"✓ Tables saved to: {TABLES_DIR}")
        print("\nNext step: Run 06_model_selection.py")
        print("="*DIVIDER_WIDTH + "\n")
        
        return models, comparison_table
    
    except Exception as e:
        print(f"\n✗ Error during degradation modeling: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    models, comparison_table = main()
