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
    ```
    git clone https://github.com/uisikdag/bootstrap_analyzer_pkg.git
    cd bootstrap_analyzer_pkg/examples
    python run_analysis_example.py
    ```

2.  **Install using pip:**
    ```
    pip install git+https://github.com/uisikdag/bootstrap_analyzer_pkg.git
    #read the Usage 
    ```
* download synthetic_bootstrap_data.csv to the same folder with Example Usage Code
* copy/paste  Example Usage Code code to an editor and save as "euc.py"

## Example Usage Code

```python

 
from bootstrap_analyzer import run_bootstrap_analysis, load_bootstrap_results
import pandas as pd # Needed to display results if desired
import os

# --- Parameters ---
INPUT_CSV = "synthetic_bootstrap_data.csv" # REQUIRED: Update this path
RESULTS_FILE = "analysis_output.pkl" # Optional: Path to save/load results

current_working_directory = os.getcwd()
full_path_csv = os.path.join(current_working_directory, INPUT_CSV)

# Define column names expected in the CSV
# IMPORTANT: Update these lists if using a different CSV file
X_COLUMNS = ['region', 'product_code', 'is_priority', 'avg_monthly_spend', 'satisfaction_score']
Y_COLUMNS = ['churn_risk', 'lifetime_value']

# --- Run Analysis ---
try:
    # Run analysis and save the results
    results = run_bootstrap_analysis(
        csv_filepath=full_path_csv,
        x_cols=X_COLUMNS,           
        y_cols=Y_COLUMNS,
        n_samples=100,           # Generate 100 bootstrap samples
        bootstrap_sample_size=None, # Use full original size for samples ; =1000 will generate data samples with 1000 rows each
        stratify_by='X',         # Stratify based on categorical features in X_COLS; 'Y','both' are other options
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
```
