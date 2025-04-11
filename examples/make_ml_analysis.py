# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 14:30:00 2025 # Adjusted time

@author: umit
"""

import os
import pandas as pd
import numpy as np
import pickle

# --- ML Libraries ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
import warnings
from sklearn.exceptions import DataConversionWarning

# Suppress DataConversionWarning often triggered by OrdinalEncoder/pipelines
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning) # Ignore potential future warnings from sklearn/pandas

# --- Function to load results (if needed) ---
def load_bootstrap_results(filepath):
    """Loads previously saved bootstrap analysis results."""
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    return results
#---------------------------------------------------------------------------------------------------------------------
RESULTS_FILE = "analysis_output.pkl" # Path where results were saved
# These columns are defined here just for initial setup/loading,
# the actual columns used will come from the loaded results.

# X_COLUMNS_DEFAULT = ['region', 'product_code', 'is_priority', 'avg_monthly_spend', 'satisfaction_score']
# Y_COLUMNS_DEFAULT = ['churn_risk', 'lifetime_value']


X_COLUMNS_DEFAULT = ['sl','sw','pl','pw']
Y_COLUMNS_DEFAULT = ['class']


# --- Configuration ---
# <<< CHOOSE ONE TARGET VARIABLE from Y_COLUMNS list >>>
# TARGET_VARIABLE = 'churn_risk'
# TARGET_VARIABLE = 'lifetime_value' # Example: Uncomment this to test regression

TARGET_VARIABLE = 'class'


# --- *** USER INPUT REQUIRED: Define Preprocessing for X Columns *** ---
# Map each column in X_COLUMNS to 'categorical', 'label', or 'numeric'
feature_processing_map = {
    # 'region': 'categorical',      # Will be OneHotEncoded (with drop='first')
    # 'product_code': 'categorical',# Will be OneHotEncoded (with drop='first')
    # 'is_priority': 'label',       # Will be LabelEncoded (0, 1, 2...)
    # 'avg_monthly_spend': 'numeric', # Will be MinMaxScaled (0-1)
    # 'satisfaction_score': 'numeric' # Will be MinMaxScaled (0-1)
    # # Add all columns from X_COLUMNS here!
    
    
    'sl': 'numeric',      
    'sw': 'numeric',       
    'pl': 'numeric',     # Will be LabelEncoded (0, 1, 2...)
    'pw': 'numeric',    # Will be MinMaxScaled (0-1)
     # Add all columns from X_COLUMNS here!
    
    
    
}
# --- End User Input ---

#-------------------------------------------------------------------------------------------------------------------
results = None
try:
    if os.path.exists(RESULTS_FILE):
        print(f"Loading results from {RESULTS_FILE}...")
        results = load_bootstrap_results(RESULTS_FILE)
        # Basic check
        if 'bootstrap_samples' not in results or not results['bootstrap_samples']:
             raise ValueError("Loaded results file does not contain valid bootstrap samples.")
        print("Results loaded.")
        # *** Use columns defined in the loaded results file ***
        X_COLUMNS = results.get('x_cols', X_COLUMNS_DEFAULT)
        Y_COLUMNS = results.get('y_cols', Y_COLUMNS_DEFAULT)
    else:
        raise FileNotFoundError(f"Results file {RESULTS_FILE} not found. Please run the analysis first.")

except Exception as e:
    print(f"Error loading or validating results: {e}")
    results = None

# --- ML Analysis with User-Defined Preprocessing ---
if results and results.get('bootstrap_samples'):
    print("\n--- Starting ML Analysis with User-Defined Preprocessing ---")

    # --- Validate User Input ---
    if TARGET_VARIABLE not in Y_COLUMNS:
        print(f"Error: Chosen target variable '{TARGET_VARIABLE}' not found in results. Available: {Y_COLUMNS}")
        results = None # Prevent further execution
    elif set(feature_processing_map.keys()) != set(X_COLUMNS):
        print("Error: The keys in 'feature_processing_map' do not exactly match the columns in X_COLUMNS found in the results.")
        print(f"Map keys: {set(feature_processing_map.keys())}")
        print(f"X_COLUMNS: {set(X_COLUMNS)}")
        results = None # Prevent further execution

if results and results.get('bootstrap_samples'): # Check results again after validation

        bootstrap_samples = results['bootstrap_samples']
        all_metrics = [] # Store either accuracy or R2 scores

        # --- Separate columns based on user map ---
        categorical_cols = [col for col, type in feature_processing_map.items() if type == 'categorical']
        label_cols = [col for col, type in feature_processing_map.items() if type == 'label']
        numeric_cols = [col for col, type in feature_processing_map.items() if type == 'numeric']

        print(f"\nTarget Variable: {TARGET_VARIABLE}")
        print(f"Categorical features (OneHotEncode, drop='first'): {categorical_cols}")
        print(f"Label features (OrdinalEncode): {label_cols}")
        print(f"Numeric features (MinMaxScaler): {numeric_cols}")


        # --- Determine Task Type (Simple Heuristic on first sample) ---
        first_sample = bootstrap_samples[0]
        target_series = first_sample[TARGET_VARIABLE]
        task_type = 'classification' # Default assumption
        if pd.api.types.is_float_dtype(target_series) or \
           (pd.api.types.is_integer_dtype(target_series) and target_series.nunique() > 10):
            task_type = 'regression'

        # --- Select Model and Metric based on Task Type ---
        if task_type == 'classification':
            # Note: LogisticRegression handles potential label encoded features fine.
            model = RandomForestClassifier(random_state=42)
            metric_func = accuracy_score
            metric_name = "Accuracy"
            print(f"\nTask Type: Classification (using Random Forest Classifier, Metric: {metric_name})")
            stratify_split = True
        else: # Regression
            model = RandomForestRegressor(random_state=42)
            metric_func = r2_score
            metric_name = "R-squared"
            print(f"\nTask Type: Regression (using Random Forest Regressor, Metric: {metric_name})")
            stratify_split = False

        # --- Create Preprocessor using ColumnTransformer ---
        transformers = []
        if categorical_cols:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols))
        if label_cols:
             # OrdinalEncoder assigns 0, 1, 2...
             # handle_unknown='use_encoded_value', unknown_value=-1 ensures test data with unseen labels doesn't crash transform
            transformers.append(('lbl', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), label_cols))
        if numeric_cols:
            transformers.append(('num', MinMaxScaler(), numeric_cols))

        # If no transformers are defined (e.g., empty X_COLUMNS or map), handle appropriately
        if not transformers:
             print("Warning: No preprocessing steps defined based on the feature map. Model will receive raw data.")
             # Option 1: Use identity transformer (less ideal if model expects processed data)
             # from sklearn.preprocessing import FunctionTransformer
             # preprocessor = FunctionTransformer(lambda x: x)
             # Option 2: Create an empty ColumnTransformer (might cause issues downstream)
             # preprocessor = ColumnTransformer(transformers=[])
             # Option 3: Raise error or handle based on specific needs
             # For this example, we will proceed, but models might fail.
             # Let's create an 'empty' preprocessor that essentially passes data through
             preprocessor = ColumnTransformer(transformers=[], remainder='passthrough')

        else:
            # remainder='drop' will drop any columns not explicitly handled by the transformers
            preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')


        # --- Create the Full Pipeline ---
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', model)])

        # --- Loop through Bootstrap Samples ---
        for i, sample_df in enumerate(bootstrap_samples):
            # Ensure sample_df contains the necessary columns, handle potential errors
            try:
                X_sample = sample_df[X_COLUMNS]
                y_sample = sample_df[TARGET_VARIABLE]
            except KeyError as e:
                print(f"  Skipping sample {i+1}: Missing column {e}")
                continue


            # Drop rows with NaNs in target variable
            if y_sample.isnull().any():
                X_sample = X_sample[y_sample.notnull()]
                y_sample = y_sample[y_sample.notnull()]

            # Skip if data is empty or (for classification) if only one class remains
            is_classification_and_single_class = (task_type == 'classification' and y_sample.nunique() < 2)
            if X_sample.empty or y_sample.empty or is_classification_and_single_class:
                continue # Skip this sample

            # Split data *within* the bootstrap sample
            X_train, X_test, y_train, y_test = train_test_split(
                X_sample, y_sample, test_size=0.3, random_state=i,
                stratify=y_sample if stratify_split else None
            )

            try:
                # Train the entire pipeline
                pipeline.fit(X_train, y_train)

                # Predict
                y_pred = pipeline.predict(X_test)

                # Calculate the selected metric
                metric_value = metric_func(y_test, y_pred)
                all_metrics.append(metric_value)

            except ValueError as ve:
                 # Catch potential errors during fit/predict, e.g., from OrdinalEncoder unknown values if not handled
                 print(f"  Skipping sample {i+1} due to ValueError during pipeline processing: {ve}")
                 all_metrics.append(np.nan)
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
            print(f"Could not calculate valid {metric_name} for any bootstrap sample.")

else:
    print("\nSkipping Machine Learning Analysis: No valid bootstrap results found or loaded.")