"""
================================================================================
CONFIGURATION
================================================================================

Centralized configuration for the entire CALCE CS2-35 methodology implementation.

All parameters are defined here to ensure reproducibility and easy modification.

DO NOT hard-code values in individual scripts.

================================================================================
"""

import os

# ============================================================================
# PATHS
# ============================================================================

DATASET_PATH = r"C:\Users\catar\OneDrive\Área de Trabalho\MESTRADO\TESTE_METODOLOGIA\CS2_35"

OUTPUT_BASE_DIR = r"C:\Users\catar\OneDrive\Área de Trabalho\MESTRADO\TESTE_METODOLOGIA"

# Create subdirectories for organized output
FIGURES_DIR = os.path.join(OUTPUT_BASE_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_BASE_DIR, "tables")
DATA_DIR = os.path.join(OUTPUT_BASE_DIR, "processed_data")
LOGS_DIR = os.path.join(OUTPUT_BASE_DIR, "logs")

# Create directories if they don't exist
for directory in [FIGURES_DIR, TABLES_DIR, DATA_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# METHODOLOGICAL PARAMETERS
# ============================================================================

# Model validation strategy
TRAIN_FRACTION = 0.70  # First 70% of observations for training

# Statistical significance level
ALPHA = 0.05

# Functional failure criterion (adopted conventional criterion for this research)
SOH_FAILURE_THRESHOLD = 80.0  # Percentage

# Persistence criterion for Potential Failure detection
PERSISTENCE_WINDOW = 5  # Number of observations

# Polynomial degrees to evaluate for candidate degradation models
POLYNOMIAL_DEGREES = [2, 3, 4]

# ============================================================================
# RANDOM SEED FOR REPRODUCIBILITY
# ============================================================================

RANDOM_SEED = 42

# ============================================================================
# FIGURE AND TABLE FORMATS
# ============================================================================

# Figure DPI for publication quality
FIGURE_DPI = 300

# Figure format
FIGURE_FORMAT = 'png'

# Font sizes
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_TICK = 10
FONT_SIZE_LEGEND = 10

# Line width for plots
LINE_WIDTH = 2.0
LINE_WIDTH_THIN = 1.0

# Marker size for scatter plots
MARKER_SIZE = 6

# ============================================================================
# TABLE EXPORT FORMATS
# ============================================================================

# Export tables as CSV and Excel for maximum compatibility
TABLE_FORMATS = ['csv', 'excel']

# ============================================================================
# VALIDATION SETTINGS
# ============================================================================

# Minimum number of observations required for reliable analysis
MIN_OBSERVATIONS = 20

# Maximum extrapolation distance (cycles beyond observed data)
# If failure cycle prediction requires extrapolation beyond this,
# a warning will be issued
MAX_EXTRAPOLATION_DISTANCE = 100

# ============================================================================
# PRINT CONFIGURATION
# ============================================================================

# Set to True for verbose output
VERBOSE = True

# Width of output section dividers
DIVIDER_WIDTH = 80
