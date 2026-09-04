# CALCE CS2-35 Battery Degradation Analysis

Comprehensive methodology for lithium-ion battery degradation analysis using the CALCE CS2-35 dataset.

## Overview

This analysis implements a rigorous 10-step methodology for:
- Battery health monitoring using State of Health (SoH) calculations
- Statistical degradation modeling using multiple candidate models
- Model validation through temporal cross-validation
- Uncertainty quantification for predictions
- Potential Failure detection using Pettitt's change-point test
- Remaining Useful Life (RUL) estimation
- P-F interval calculation

## Key Features

✓ **Automated Data Pipeline**: Data loading → Processing → SoH calculation → Analysis

✓ **Modular Architecture**: Each step can run independently or as part of the complete pipeline

✓ **Multiple Degradation Models**: Linear, Polynomial (degrees 2-4), Exponential, Logarithmic, Power-Law

✓ **Rigorous Validation**: Temporal cross-validation (70% train / 30% test)

✓ **Comprehensive Diagnostics**: Residual analysis, normality testing, heteroscedasticity assessment

✓ **Uncertainty Quantification**: Prediction intervals with extrapolation analysis

✓ **Change-Point Detection**: Pettitt test for identifying Potential Failure with persistence criterion

✓ **Publication-Quality Outputs**: High-resolution figures and detailed summary tables

## Directory Structure

```
mestrado-ita-battery-degradation/
├── config.py                                  # Centralized configuration
├── 00_dataset_inspection.py                  # Step 1: Dataset inspection
├── 01_load_data.py                           # Step 2: Data loading
├── 02_data_processing.py                     # Step 3: Data processing
├── 03_soh_calculation.py                     # Step 4: SoH calculation
├── 04_exploratory_analysis.py                # Step 5: Exploratory analysis
├── 05_degradation_models.py                  # Step 6: Model fitting
├── 06_residual_analysis_and_validation.py   # Step 7: Validation
├── 07_uncertainty_quantification.py          # Step 8: Uncertainty
├── 08_pettitt_change_point.py               # Step 9: Change-point detection
├── 09_failure_analysis_and_rul.py           # Step 10: Failure analysis
├── 10_final_summary.py                       # Final summary report
├── run_analysis.py                           # Master execution script
└── README.md                                 # This file

Output directories (auto-created):
├── figures/                                  # Publication-quality figures
├── tables/                                   # Summary tables (CSV format)
├── processed_data/                           # Intermediate data files
└── logs/                                     # Execution logs (if needed)
```

## Methodology Overview

### Step 1: Dataset Inspection
- Identify all data files in the dataset directory
- Determine file formats and directory structure
- List contents and file sizes
- Prepare for automated loading

### Step 2: Data Loading
- Load all data files (CSV, Excel, JSON, TXT)
- Handle multiple file formats and delimiters
- Consolidate data if distributed across files
- Preserve chronological order

### Step 3: Data Processing
- Automatically identify relevant columns (battery ID, cycle, capacity)
- No assumptions: searches for keywords in actual column names
- Assess data quality: missing values, duplicates, anomalies
- Create standardized dataset: battery_id, cycle, discharge_capacity

### Step 4: State of Health (SoH) Calculation
- Calculate SoH using first valid discharge capacity as reference
- SoH_i = (Q_i / Q_ref) × 100, where Q_ref = Q_1
- Each battery normalized independently (NOT by population mean)
- Generate discharge capacity and SoH vs cycle plots

### Step 5: Exploratory Degradation Analysis
- Descriptive statistics (mean, std, min, max, median)
- Degradation rate analysis (cycle-to-cycle changes)
- Monotonicity assessment (verify degradation is monotonic)
- Identify nonlinear behavior and rate changes
- Generate exploratory visualizations

### Step 6: Statistical Degradation Modeling
- Fit 6 candidate models:
  - Linear: SoH(N) = β₀ + β₁·N
  - Polynomial (degrees 2, 3, 4)
  - Exponential: SoH(N) = β₀·exp(-β₁·N)
  - Logarithmic: SoH(N) = β₀ - β₁·ln(N)
  - Power-Law: SoH(N) = β₀ - β₁·N^β₂

- Calculate metrics:
  - R² and Adjusted R²
  - RMSE and MAE
  - Residual standard deviation
  - Number of parameters (parsimony)

### Step 7: Residual Analysis & Temporal Validation
- **Temporal Validation** (70% training, 30% validation):
  - Preserve chronological order
  - NO random shuffling
  - Fit model on training data only
  - Evaluate on held-out test data

- **Residual Diagnostics**:
  - Residuals vs Cycle
  - Residuals vs Fitted Values
  - Histogram of residuals
  - Q-Q plot against normal distribution

- **Shapiro-Wilk Normality Test**:
  - Test if residuals follow normal distribution
  - p-value < 0.05 → reject normality assumption
  - Used ONLY for assessing model adequacy
  - NOT used for detecting Potential Failure

### Step 8: Uncertainty Quantification
- Calculate residual standard deviation (s_e)
- Construct 95% prediction intervals
- Account for both model uncertainty and measurement variability
- Analyze how uncertainty grows with extrapolation
- Distinguish predictions from observed data

### Step 9: Pettitt Change-Point Detection
- **Pettitt Test** (non-parametric):
  - Applied to RESIDUALS (not raw SoH)
  - Detects shift in residual behavior
  - p < 0.05 indicates significant change point

- **Persistence Criterion**:
  - Change must persist over 5 subsequent observations
  - Evaluates whether detected change is robust
  - Both statistical significance AND persistence required
  - If either fails → no Potential Failure detected

**Important Distinctions**:
- Potential Failure ≠ physical onset of degradation
- Potential Failure ≠ arbitrary SoH threshold
- Potential Failure = detectable change in degradation pattern

### Step 10: Failure Analysis & RUL
- **Functional Failure**:
  - Criterion: SoH reaches 80% (conventional for this research)
  - Estimated by interpolation or extrapolation
  - Cycle at which N_F_hat occurs

- **Remaining Useful Life (RUL)**:
  - RUL(N_c) = N_F_hat - N_c
  - Calculated from current cycle to predicted failure
  - Decreases monotonically with advancing cycles

- **P-F Interval**:
  - If Potential Failure detected: P-F = N_F - N_P
  - Interval between Potential and Functional Failure
  - Indicates time available for intervention
  - If no Potential Failure: not calculated

## Usage

### Run All Steps
```bash
python run_analysis.py --all
```

### Run Single Step
```bash
python run_analysis.py 1          # Run step 1 only
python run_analysis.py 4          # Run step 4 only
```

### List Available Steps
```bash
python run_analysis.py --list
```

### Generate Summary Report
```bash
python 10_final_summary.py
```

## Configuration

Edit `config.py` to customize:
- Dataset path
- Output directories
- Analysis parameters:
  - `TRAIN_FRACTION`: Training data fraction (default 0.70)
  - `ALPHA`: Significance level (default 0.05)
  - `SOH_FAILURE_THRESHOLD`: Failure criterion (default 80%)
  - `PERSISTENCE_WINDOW`: Persistence window (default 5 observations)
  - `POLYNOMIAL_DEGREES`: Polynomial degrees to test (default [2, 3, 4])

## Output Files

### Figures (figures/ directory)
- `01_discharge_capacity_vs_cycle.png` - Raw capacity degradation
- `02_soh_vs_cycle.png` - Normalized SoH degradation
- `03_degradation_rate_variation.png` - Cycle-to-cycle rate changes
- `04_soh_distribution.png` - SoH value distribution
- `05_candidate_models_comparison.png` - All fitted models
- `06-09_*.png` - Residual diagnostic plots
- `10_temporal_validation_*.png` - Validation results
- `11_prediction_intervals_*.png` - Model uncertainty
- `12_pettitt_change_point_detection.png` - Change-point analysis
- `13_failure_analysis.png` - Failure cycle estimation
- `14_rul_vs_cycle.png` - RUL projection

### Tables (tables/ directory)
- `Table_01_SoH_Summary.csv` - SoH calculations per battery
- `Table_02_Descriptive_Statistics.csv` - Degradation statistics
- `Table_03_Degradation_Rates.csv` - Rate analysis
- `Table_04_Monotonicity_Assessment.csv` - Monotonicity check
- `Table_05_Model_Comparison.csv` - Model fit metrics
- `Table_06_Shapiro_Wilk_Normality_Test.csv` - Residual normality
- `Table_07_Temporal_Validation_Metrics.csv` - Validation performance
- `Table_08_Uncertainty_Metrics.csv` - Prediction uncertainty
- `Table_09_Pettitt_Change_Point_Detection.csv` - Change-point results
- `Table_10_Failure_and_RUL_Analysis.csv` - Failure and RUL estimates

### Data (processed_data/ directory)
- `processed_data.csv` - Standardized data (battery_id, cycle, discharge_capacity)
- `soh_data.csv` - SoH calculations
- `fitted_models.pkl` - Pickled fitted models
- `battery_data_info.pkl` - Battery metadata

## Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
scikit-learn>=0.24.0  (optional, for advanced analysis)
```

## Key Methodological Principles

1. **Do NOT assume column names** - Automatically search for keywords
2. **Preserve chronological order** - Maintain temporal structure
3. **Individual normalization** - Each battery independently (Q_ref = Q_1)
4. **Multiple candidate models** - Avoid premature model selection
5. **Residual focus** - Assess fit quality through residual behavior
6. **Temporal validation** - 70% train / 30% test chronologically split
7. **Statistical significance** - Both p-value AND persistence required
8. **Explicit uncertainty** - Quantify prediction intervals
9. **Distinguish metrics** - RUL ≠ P-F interval
10. **Document assumptions** - Explicit about what's conventional vs universal

## References

- CALCE Battery Research Group: http://calce.umd.edu/
- Pettitt, A.N. (1979). "A Non-Parametric Approach to the Change-Point Problem." Journal of the Royal Statistical Society.
- IEEE 1625-2008: Standard for Rechargeable Batteries for Use in Notebook Computers

## Author

Catarina Claudino
Master's Dissertation - ITA (Instituto Tecnológico de Aeronáutica)

## License

MIT License

