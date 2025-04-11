# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 10:10:23 2025

@author: umit
"""

# Bootstrap Analyzer

A Python package for analyzing data features and generating (stratified) bootstrap samples from tabular data stored in CSV files.

## Features

* Reads data from CSV files.
* Automatically detects categorical and numerical features based on data type and unique value counts.
* Generates N bootstrap samples (sampling with replacement).
* Supports stratified bootstrapping based on categorical independent (X), dependent (Y), or both sets of variables.
* Validates requested bootstrap sample size against original data size and smallest stratum size (when stratifying).
* Optionally saves analysis results (including bootstrap samples) to a file using Pickle.
* Provides a function to load saved results.

## Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone [https://github.com/yourusername/bootstrap_analyzer.git](https://github.com/yourusername/bootstrap_analyzer.git) # Replace with actual URL if hosted
    cd bootstrap_analyzer_pkg
    ```

2.  **Install using pip:**
    It's recommended to install in a virtual environment.

    ```bash
    # Create and activate a virtual environment (optional)
    # python -m venv venv
    # source venv/bin/activate  # On Windows use `venv\Scripts\activate`

    # Install the package
    pip install .
    ```
    This command reads `setup.py` and installs the package and its dependencies listed in `requirements.txt`.

## Usage

```python
from bootstrap_analyzer import run_bootstrap_analysis, load_bootstrap_results
import pandas as pd # Needed to display results if desired

# --- Parameters ---
INPUT_CSV = "path/to/your/data.csv" # REQUIRED: Update this path
RESULTS_FILE = "analysis_output.pkl" # Optional: Path to save/load results

# Define which columns are independent (X) and dependent (Y) variables
# REQUIRED: Update these lists to match your data.csv
X_COLS = ['feature1', 'feature2_cat', 'feature3_num']
Y_COLS = ['target_variable', 'other_outcome']

# --- Run Analysis ---
try:
    # Run analysis and save the results
    results = run_bootstrap_analysis(
        csv_filepath=INPUT_CSV,
        x_cols=X_COLS,
        y_cols=Y_COLS,
        n_samples=100,           # Generate 100 bootstrap samples
        bootstrap_sample_size=None, # Use full original size for samples
        stratify_by='X',         # Stratify based on categorical features in X_COLS
        random_state=42,         # For reproducible results
        save_results_path=RESULTS_FILE # Save the output
    )

    # --- Process Results (Example) ---
    if results:
        print("Analysis successful!")
        print("\nFeature Types Detected:")
        print(results['feature_types'])

        print("\nCategory Counts:")
        print(results['category_counts'])

        print(f"\nNumber of bootstrap samples generated: {len(results['bootstrap_samples'])}")
        if results['bootstrap_samples']:
             print("Head of the first bootstrap sample:")
             # Ensure pandas is imported to use .head()
             print(results['bootstrap_samples'][0].head())

except Exception as e:
    print(f"An error occurred: {e}")


# --- Load Previously Saved Results (Example) ---
# try:
#     loaded_results = load_bootstrap_results(RESULTS_FILE)
#     print("\nSuccessfully loaded previous results.")
#     print(f"Number of samples in loaded results: {len(loaded_results['bootstrap_samples'])}")
# except Exception as e:
#     print(f"Failed to load results: {e}")