"""
================================================================================
MASTER EXECUTION SCRIPT
================================================================================

This script orchestrates the complete analysis pipeline for the CALCE CS2-35
battery degradation dataset.

Execution order:
  1. Dataset inspection
  2. Data loading
  3. Data processing and column identification
  4. State of Health (SoH) calculation
  5. Exploratory degradation analysis
  6. Statistical degradation modeling (candidate models)
  7. Residual analysis and temporal validation
  8. Uncertainty quantification
  9. Pettitt change-point detection
  10. Failure analysis, RUL, and P-F interval
  11. Generate final comprehensive summary

Each step is independent and can be run separately.
For full analysis, execute steps in order.

================================================================================
"""

import os
import sys
import subprocess
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_BASE_DIR, VERBOSE

# ============================================================================
# CONFIGURATION
# ============================================================================

STEPS = [
    {
        'number': 1,
        'name': 'Dataset Inspection',
        'script': '00_dataset_inspection.py',
        'description': 'Inspect CALCE dataset structure and identify files'
    },
    {
        'number': 2,
        'name': 'Data Loading',
        'script': '01_load_data.py',
        'description': 'Load all data files into memory'
    },
    {
        'number': 3,
        'name': 'Data Processing',
        'script': '02_data_processing.py',
        'description': 'Process data, identify columns, assess quality'
    },
    {
        'number': 4,
        'name': 'SoH Calculation',
        'script': '03_soh_calculation.py',
        'description': 'Calculate capacity-based State of Health'
    },
    {
        'number': 5,
        'name': 'Exploratory Analysis',
        'script': '04_exploratory_analysis.py',
        'description': 'Exploratory degradation analysis'
    },
    {
        'number': 6,
        'name': 'Degradation Modeling',
        'script': '05_degradation_models.py',
        'description': 'Fit candidate degradation models'
    },
    {
        'number': 7,
        'name': 'Residual Analysis & Validation',
        'script': '06_residual_analysis_and_validation.py',
        'description': 'Analyze residuals and validate model'
    },
    {
        'number': 8,
        'name': 'Uncertainty Quantification',
        'script': '07_uncertainty_quantification.py',
        'description': 'Quantify model prediction uncertainty'
    },
    {
        'number': 9,
        'name': 'Pettitt Change-Point Detection',
        'script': '08_pettitt_change_point.py',
        'description': 'Detect Potential Failure using Pettitt test'
    },
    {
        'number': 10,
        'name': 'Failure Analysis & RUL',
        'script': '09_failure_analysis_and_rul.py',
        'description': 'Analyze failure, calculate RUL and P-F interval'
    }
]

# ============================================================================
# EXECUTION FUNCTIONS
# ============================================================================

def print_header():
    """Print welcome header."""
    print(f"\n{'='*80}")
    print("CALCE CS2-35 BATTERY DEGRADATION ANALYSIS")
    print("Master Execution Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

def print_step_info(step):
    """Print information about a step."""
    print(f"\n{'='*80}")
    print(f"STEP {step['number']}: {step['name'].upper()}")
    print(f"{'='*80}")
    print(f"Description: {step['description']}")
    print(f"Script: {step['script']}")
    print(f"{'='*80}\n")

def run_step(step):
    """
    Execute a single analysis step.
    
    Parameters:
    -----------
    step : dict
        Step configuration dictionary
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    
    print_step_info(step)
    
    script_path = os.path.join(os.path.dirname(__file__), step['script'])
    
    if not os.path.exists(script_path):
        print(f"✗ ERROR: Script not found: {script_path}")
        return False
    
    try:
        print(f"Executing: {step['script']}...")
        start_time = time.time()
        
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            text=True,
            cwd=os.path.dirname(__file__)
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✓ Step {step['number']} completed successfully")
            print(f"  Elapsed time: {elapsed:.1f} seconds")
            return True
        else:
            print(f"\n✗ Step {step['number']} failed with return code {result.returncode}")
            return False
    
    except Exception as e:
        print(f"\n✗ Error executing step {step['number']}: {e}")
        return False

def run_all_steps():
    """
    Execute all steps in the analysis pipeline.
    
    Returns:
    --------
    dict
        Summary of execution results
    """
    
    results = {
        'total_steps': len(STEPS),
        'completed_steps': 0,
        'failed_steps': [],
        'start_time': datetime.now(),
        'end_time': None
    }
    
    for step in STEPS:
        if run_step(step):
            results['completed_steps'] += 1
        else:
            results['failed_steps'].append(step['number'])
    
    results['end_time'] = datetime.now()
    return results

def print_summary(results):
    """
    Print execution summary.
    
    Parameters:
    -----------
    results : dict
        Execution results
    """
    
    duration = results['end_time'] - results['start_time']
    minutes = duration.total_seconds() / 60
    
    print(f"\n{'='*80}")
    print("ANALYSIS PIPELINE EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"\nExecution Time:")
    print(f"  Start: {results['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  End: {results['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration: {minutes:.1f} minutes")
    
    print(f"\nResults:")
    print(f"  Total steps: {results['total_steps']}")
    print(f"  Completed: {results['completed_steps']}")
    print(f"  Failed: {len(results['failed_steps'])}")
    
    if results['failed_steps']:
        print(f"\n  Failed steps: {', '.join(map(str, results['failed_steps']))}")
    else:
        print(f"\n  ✓ ALL STEPS COMPLETED SUCCESSFULLY")
    
    print(f"\nOutput Directory: {OUTPUT_BASE_DIR}")
    print(f"Figures: {os.path.join(OUTPUT_BASE_DIR, 'figures')}")
    print(f"Tables: {os.path.join(OUTPUT_BASE_DIR, 'tables')}")
    print(f"Data: {os.path.join(OUTPUT_BASE_DIR, 'processed_data')}")
    
    print(f"\n{'='*80}\n")

def print_step_list():
    """
    Print list of available steps.
    """
    print(f"\n{'='*80}")
    print("AVAILABLE ANALYSIS STEPS")
    print(f"{'='*80}\n")
    
    for step in STEPS:
        print(f"{step['number']:2d}. {step['name']:30s} - {step['description']}")
    
    print(f"\n{'='*80}\n")

def run_single_step(step_number):
    """
    Execute a single step by number.
    
    Parameters:
    -----------
    step_number : int
        Step number to execute (1-indexed)
    """
    
    step_number = int(step_number)
    
    if step_number < 1 or step_number > len(STEPS):
        print(f"Error: Step number must be between 1 and {len(STEPS)}")
        return
    
    step = STEPS[step_number - 1]
    
    results = {
        'total_steps': 1,
        'start_time': datetime.now(),
        'end_time': None,
        'completed_steps': 0,
        'failed_steps': []
    }
    
    if run_step(step):
        results['completed_steps'] = 1
    else:
        results['failed_steps'] = [step_number]
    
    results['end_time'] = datetime.now()
    print_summary(results)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    """
    
    print_header()
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg == '--list':
            print_step_list()
        elif arg.isdigit():
            print(f"Running step {arg}...")
            run_single_step(arg)
        elif arg == '--all':
            print("Running all steps...")
            results = run_all_steps()
            print_summary(results)
        else:
            print(f"Unknown argument: {arg}")
            print("\nUsage:")
            print(f"  python {os.path.basename(__file__)} --list        (show available steps)")
            print(f"  python {os.path.basename(__file__)} <step_num>    (run single step)")
            print(f"  python {os.path.basename(__file__)} --all         (run all steps)")
    else:
        print("No argument provided.")
        print("\nUsage:")
        print(f"  python {os.path.basename(__file__)} --list        (show available steps)")
        print(f"  python {os.path.basename(__file__)} <step_num>    (run single step)")
        print(f"  python {os.path.basename(__file__)} --all         (run all steps)")
        print("\nExample:")
        print(f"  python {os.path.basename(__file__)} 1             (run step 1)")
        print(f"  python {os.path.basename(__file__)} --all         (run all 10 steps)")

if __name__ == "__main__":
    main()
