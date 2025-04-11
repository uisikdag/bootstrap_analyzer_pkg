# Bootstrap Analyzer

A Python package for generating (stratified) bootstrap samples from tabular data stored in CSV files.Stratificarion
is done using categories of 'X' columns or 'Y' columns or 'both'. The sample size can be smaller than the data size,number
of samples can be configured.

## Features

* Reads data from CSV files.
* Automatically detects categorical and numerical features based on data type and unique value counts.
* Generates N bootstrap samples (sampling with replacement).
* Supports stratified bootstrapping based on categorical independent (X), dependent (Y), or both sets of variables.
* Validates requested bootstrap sample size against original data size and smallest stratum size (when stratifying).
* Optionally saves analysis results (including bootstrap samples) to a file using Pickle.
* Provides a function to load saved results.

## Installation

1.  Option A: **Install using Clone**
    ```
    git clone https://github.com/uisikdag/bootstrap_analyzer_pkg.git
    cd bootstrap_analyzer_pkg/examples
    python run_analysis_example.py
    python make_ml_analysis.py
    ```

2.  Option B: **Install using pip:**

 <br>Step A:
     ```
    pip install git+https://github.com/uisikdag/bootstrap_analyzer_pkg.git
    ```
<br>Step B:     
* download synthetic_bootstrap_data.csv to the same folder with the code below
* copy/paste the code below to an editor > save as "rae.py" > python rae.py

```python
from bootstrap_analyzer import run_bootstrap_analysis
import os

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

# --- Run Analysis ---
try:
    # Run analysis and save the results
    results = run_bootstrap_analysis(
        csv_filepath=full_path_csv,
        x_cols=X_COLUMNS,           
        y_cols=Y_COLUMNS,
        n_samples=50,           # Generate 50 bootstrap samples
        bootstrap_sample_size=100, # generate bootstrap samples with 100 rows each; None: Use full original size of input data for the samples ;
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
<br> Step C:
* copy/paste the code below to an editor> save as "mma.py" > pythob mma.py

```
# --- Assume previous code ran and produced 'results' ---
# --- Or load 'results' if available ---

import os
import pandas as pd
import numpy as np
import pickle

# --- Function to load results (if needed) ---
def load_bootstrap_results(filepath):
    """Loads previously saved bootstrap analysis results."""
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    return results

RESULTS_FILE = "analysis_output.pkl" # Path where results were saved
X_COLUMNS = ['region', 'product_code', 'is_priority', 'avg_monthly_spend', 'satisfaction_score'] # Define expected columns
Y_COLUMNS = ['churn_risk', 'lifetime_value'] # Define expected columns

results = None
try:
    if os.path.exists(RESULTS_FILE):
        print(f"Loading results from {RESULTS_FILE}...")
        results = load_bootstrap_results(RESULTS_FILE)
        # Basic check
        if 'bootstrap_samples' not in results or not results['bootstrap_samples']:
             raise ValueError("Loaded results file does not contain valid bootstrap samples.")
        print("Results loaded.")
        # Use columns from loaded results if they exist, otherwise use script defaults
        X_COLUMNS = results.get('x_cols', X_COLUMNS)
        Y_COLUMNS = results.get('y_cols', Y_COLUMNS)
    else:
        # If the results file doesn't exist, you would need to run the
        # bootstrap_analyzer.run_bootstrap_analysis part here first.
        # For this example, we'll assume the file MUST exist.
        raise FileNotFoundError(f"Results file {RESULTS_FILE} not found. Please run the analysis first.")

except Exception as e:
    print(f"Error loading or validating results: {e}")
    # Exit or handle error appropriately if results are needed
    results = None

# --- Simplified ML Analysis with Auto Task Selection ---
if results and results.get('bootstrap_samples'):
    print("\n--- Starting Simplified ML Analysis with Auto Task Selection ---")

    # --- Configuration ---
    # <<< CHOOSE ONE TARGET VARIABLE from Y_COLUMNS list >>>
    TARGET_VARIABLE = 'lifetime_value'
    #TARGET_VARIABLE = 'churn_risk' # Example: Uncomment this to test classification

    if TARGET_VARIABLE not in Y_COLUMNS:
        print(f"Error: Chosen target variable '{TARGET_VARIABLE}' not found in results. Available: {Y_COLUMNS}")
    else:
        # --- ML Imports ---
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.metrics import accuracy_score, r2_score # Include R2 score
        import numpy as np
        import pandas as pd # Ensure pandas is imported

        bootstrap_samples = results['bootstrap_samples']
        all_metrics = [] # Store either accuracy or R2 scores

        # --- Identify Feature Types (using the first sample) ---
        first_sample = bootstrap_samples[0]
        categorical_features = first_sample[X_COLUMNS].select_dtypes(exclude=np.number).columns.tolist()
        numerical_features = first_sample[X_COLUMNS].select_dtypes(include=np.number).columns.tolist()

        print(f"Target Variable: {TARGET_VARIABLE}")
        print(f"Categorical Features for OneHotEncoding: {categorical_features}")
        print(f"Numerical Features (passed through): {numerical_features}")

        # --- Determine Task Type (Simple Heuristic) ---
        target_series = first_sample[TARGET_VARIABLE]
        task_type = 'classification' # Default assumption
        # If target is float OR if it's integer type with many unique values, assume regression
        if pd.api.types.is_float_dtype(target_series) or \
           (pd.api.types.is_integer_dtype(target_series) and target_series.nunique() > 10): # Heuristic: >10 unique ints -> regression
            task_type = 'regression'

        # --- Select Model and Metric based on Task Type ---
        if task_type == 'classification':
            model = LogisticRegression(solver='liblinear', random_state=42)
            metric_func = accuracy_score
            metric_name = "Accuracy"
            print(f"Task Type: Classification (using LogisticRegression, Metric: {metric_name})")
            # Stratify only makes sense for classification
            stratify_split = True
        else: # Regression
            model = LinearRegression()
            metric_func = r2_score
            metric_name = "R-squared"
            print(f"Task Type: Regression (using LinearRegression, Metric: {metric_name})")
            stratify_split = False

        # --- Simple Preprocessor (Same for both tasks in this simple example) ---
        # Only apply OneHotEncoder to categorical features, pass others through
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
            remainder='passthrough' # Keep numerical columns
        )

        # --- Loop through Bootstrap Samples ---
        for i, sample_df in enumerate(bootstrap_samples):
            X_sample = sample_df[X_COLUMNS]
            y_sample = sample_df[TARGET_VARIABLE]

            # Drop rows with NaNs in target variable for this specific model run
            if y_sample.isnull().any():
                X_sample = X_sample[y_sample.notnull()]
                y_sample = y_sample[y_sample.notnull()]

            # Skip if data is empty or (for classification) if only one class remains
            is_classification_and_single_class = (task_type == 'classification' and y_sample.nunique() < 2)
            if X_sample.empty or y_sample.empty or is_classification_and_single_class:
                # print(f"  Skipping sample {i+1} due to empty data or single class in target.")
                continue # Skip this sample

            # Split data *within* the bootstrap sample
            X_train, X_test, y_train, y_test = train_test_split(
                X_sample, y_sample, test_size=0.3, random_state=i,
                stratify=y_sample if stratify_split else None # Stratify only if needed
            )

            try:
                # Apply preprocessing
                X_train_processed = preprocessor.fit_transform(X_train)
                X_test_processed = preprocessor.transform(X_test)

                # Train the selected model
                model.fit(X_train_processed, y_train)

                # Predict
                y_pred = model.predict(X_test_processed)

                # Calculate the selected metric
                metric_value = metric_func(y_test, y_pred)
                all_metrics.append(metric_value)

            except Exception as e:
                print(f"  Error processing sample {i+1}: {e}")
                all_metrics.append(np.nan) # Append NaN if an error occurred

        # --- Calculate Average Performance ---
        valid_metrics = [m for m in all_metrics if not np.isnan(m)]
        if valid_metrics:
            average_metric = np.mean(valid_metrics)
            std_dev_metric = np.std(valid_metrics)
            print("\n--- ML Result ---")
            print(f"Average {metric_name} for '{TARGET_VARIABLE}' ({task_type}) across {len(valid_metrics)} bootstrap samples: {average_metric:.4f}")
            print(f"Standard Deviation of {metric_name}: {std_dev_metric:.4f}")
        else:
            print("\n--- ML Result ---")
            print(f"Could not calculate {metric_name} for any bootstrap sample.")

else:
    print("\nSkipping Machine Learning Analysis: No valid bootstrap results found or loaded.")
```
