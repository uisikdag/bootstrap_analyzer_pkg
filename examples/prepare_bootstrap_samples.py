# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 10:11:22 2025

@author: umit
"""



# examples/run_analysis_example.py

# Import functions from the installed package
from bootstrap_analyzer import run_bootstrap_analysis, load_bootstrap_results
import pandas as pd # Import pandas to work with results if needed
import os # To construct file paths relative to this script

# --- Parameters ---
INPUT_CSV = "synthetic_bootstrap_data.csv" # REQUIRED: Update this path
RESULTS_FILE = "analysis_output.pkl" # Optional: Path to save/load results

# --- Path Settings ---
current_working_directory = os.getcwd()
full_path_csv = os.path.join(current_working_directory, INPUT_CSV)

# Define column names expected in the CSV
# IMPORTANT: Update these lists if using a different CSV file
X_COLUMNS = ['region', 'product_code', 'is_priority', 'avg_monthly_spend', 'satisfaction_score']
Y_COLUMNS = ['churn_risk', 'lifetime_value']


# Bootstrap settings
NUM_BOOTSTRAP_SAMPLES = 50
BOOTSTRAP_SIZE = 100      # Custom size for samples
STRATIFICATION_METHOD = 'X'
RANDOM_SEED = 123

print("--- Starting Bootstrap Analysis Example ---")
print(f"Input CSV: {INPUT_CSV}")
print(f"Results File: {RESULTS_FILE}")

# Check if input CSV exists
if not os.path.exists(full_path_csv):
    print(f"\nError: Input CSV file not found at '{INPUT_CSV}'.")
    print("Please create the 'synthetic_bootstrap_data.csv' file (e.g., using code from previous steps) in the project root directory or update the INPUT_CSV path.")
    exit()

try:
    # --- Run Analysis & Save Results ---
    print("\nRunning bootstrap analysis...")
    results = run_bootstrap_analysis(
        csv_filepath=INPUT_CSV,
        num_y_cols=len(Y_COLUMNS),  # Specify number of Y columns instead of listing them
        n_samples=NUM_BOOTSTRAP_SAMPLES,
        bootstrap_sample_size=BOOTSTRAP_SIZE,
        stratify_by=STRATIFICATION_METHOD,
        random_state=RANDOM_SEED,
        save_results_path=RESULTS_FILE
    )

    # --- Process Results (Example) ---
    if results:
        print("\n--- Analysis Results Summary ---")
        print("\nDetected Feature Types:")
        # Pretty print
        max_len = max(len(k) for k in results['feature_types']) if results['feature_types'] else 0
        for key, val in sorted(results['feature_types'].items()):
            print(f"  {key:<{max_len}} : {val}")

        print("\nCategory Counts:")
        max_len = max(len(k) for k in results['category_counts']) if results['category_counts'] else 0
        for key, val in sorted(results['category_counts'].items()):
            print(f"  {key:<{max_len}} : {val}")

        n_generated = len(results['bootstrap_samples'])
        print(f"\nNumber of bootstrap samples generated: {n_generated}")
        if n_generated > 0:
             sample_size = len(results['bootstrap_samples'][0])
             print(f"Size of each bootstrap sample: {sample_size}")
             print("\nHead of the first bootstrap sample:")
             print(results['bootstrap_samples'][0].head())
        print(f"\nResults were saved to: {RESULTS_FILE}")

except Exception as e:
    print(f"\n--- An error occurred during analysis ---")
    print(f"{type(e).__name__}: {e}")


# --- Example: Load Results ---
print("\n--- Loading Saved Results Example ---")
try:
    loaded_results = load_bootstrap_results(RESULTS_FILE)
    print(f"\nSuccessfully loaded results from {RESULTS_FILE}.")
    print(f"Number of samples in loaded results: {len(loaded_results['bootstrap_samples'])}")
    # You can now work with 'loaded_results'
    # print(loaded_results['feature_types'])

except Exception as e:
    print(f"Failed to load results: {e}")


print("\n--- Example Script Finished ---")
