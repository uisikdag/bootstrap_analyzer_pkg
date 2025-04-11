# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 14:15:00 2025 # Adjusted time

@author: umit
"""

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