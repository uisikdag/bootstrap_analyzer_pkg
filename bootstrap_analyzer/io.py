# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 10:09:03 2025

@author: umit
"""

import pandas as pd
import numpy as np
import pickle
from typing import List, Dict, Union, Optional, Any

# Import the core function from the same package
from .core import analyze_and_bootstrap_v3

# ----------------------------------------------------------
# Wrapper Function to Run Analysis from CSV
# (Determines X/Y columns based on the number of trailing Y columns specified)
# ----------------------------------------------------------
def run_bootstrap_analysis(
    csv_filepath: str,
    num_y_cols: int, # New parameter: Number of columns from the end to treat as Y
    n_samples: int,
    bootstrap_sample_size: Optional[int] = None,
    stratify_by: str = 'X',
    categorical_threshold: int = 10,
    random_state: Optional[int] = None,
    read_csv_options: Optional[Dict[str, Any]] = None,
    save_results_path: Optional[str] = None
) -> Optional[Dict[str, Union[Dict, List[pd.DataFrame]]]]:
    """
    Reads data from a CSV file, identifies Y variables as the last 'num_y_cols'
    columns and X variables as the preceding columns from the header, runs the
    bootstrap analysis, and optionally saves results.

    Args:
        csv_filepath: Path to the input CSV file.
        num_y_cols: The number of columns, counting from the end of the file,
                    to be treated as dependent (Y) variables. Must be at least 1.
        n_samples: Number of bootstrap samples to generate.
        bootstrap_sample_size: Desired size for each bootstrap sample (optional).
                               Defaults to original data size if None.
        stratify_by: Stratification strategy ('X', 'Y', 'both', 'none').
                     Defaults to 'X'.
        categorical_threshold: Threshold for numeric features to be categorical.
        random_state: Seed for reproducibility.
        read_csv_options: Dictionary of additional kwargs for pd.read_csv.
        save_results_path: If provided, saves the results dictionary to this
                           path using pickle. Defaults to None.

    Returns:
        A dictionary containing bootstrap analysis results, or None if saving fails
        when requested.

    Raises:
        FileNotFoundError, pd.errors.EmptyDataError, Exception: From file reading.
        ValueError: If num_y_cols is invalid, CSV has too few columns, or other
                    validation issues.
        TypeError: From analyze_and_bootstrap_v3 validation.
        IOError, pickle.PicklingError: If saving results fails.
    """
    print(f"Attempting to read data from: {csv_filepath}")
    default_csv_opts = {'true_values': ['True', 'true'], 'false_values': ['False', 'false']}
    if read_csv_options: default_csv_opts.update(read_csv_options)

    try:
        df = pd.read_csv(csv_filepath, **default_csv_opts)
        print(f"Successfully read {len(df)} rows and {len(df.columns)} columns.")
        if df.empty: raise pd.errors.EmptyDataError("CSV file is empty.")
    except Exception as e:
        print(f"Error reading CSV file '{csv_filepath}': {e}")
        raise

    # --- Determine X and Y columns based on num_y_cols ---
    all_cols = df.columns.tolist()
    total_cols = len(all_cols)

    # Validate num_y_cols
    if not isinstance(num_y_cols, int) or num_y_cols < 1:
        raise ValueError(f"`num_y_cols` must be a positive integer. Received: {num_y_cols}")
    if num_y_cols >= total_cols:
        raise ValueError(f"`num_y_cols` ({num_y_cols}) cannot be greater than or equal to the total number of columns ({total_cols}). Must leave at least one X column.")

    y_cols_auto = all_cols[-num_y_cols:]  # Get the last 'num_y_cols' columns
    x_cols_auto = all_cols[:-num_y_cols]   # Get all columns before the Y columns

    print(f"Automatically detected X columns: {x_cols_auto}")
    print(f"Automatically detected Y columns ({num_y_cols} trailing): {y_cols_auto}")
    # --- End of automatic detection ---

    adjusted_sample_size = bootstrap_sample_size
    if bootstrap_sample_size is not None and bootstrap_sample_size > len(df):
         print(f"Warning: Requested bootstrap sample size ({bootstrap_sample_size}) > data size ({len(df)}). Using data size.")
         # Adjust sample size if it's larger than the dataset size
         adjusted_sample_size = len(df)

    # Use original dataframe size if bootstrap_sample_size is None
    effective_sample_size = adjusted_sample_size if adjusted_sample_size is not None else len(df)


    print(f"\nRunning bootstrap analysis with {n_samples} samples of size {effective_sample_size}...")
    # Call the core function from .core using automatically determined columns
    results = analyze_and_bootstrap_v3(
        df=df,
        x_cols=x_cols_auto,  # Use automatically determined X columns
        y_cols=y_cols_auto,  # Use automatically determined Y columns
        n_samples=n_samples,
        # Pass the potentially adjusted or original size to the core function
        bootstrap_sample_size=effective_sample_size if bootstrap_sample_size is not None else None,
        stratify_by=stratify_by,
        categorical_threshold=categorical_threshold,
        random_state=random_state
    )
    print("Bootstrap analysis completed.")

    # Save Results if path provided
    if save_results_path:
        print(f"Attempting to save analysis results to: {save_results_path}")
        try:
            with open(save_results_path, 'wb') as f:
                pickle.dump(results, f)
            print("Results saved successfully.")
        except (IOError, pickle.PicklingError) as e:
            print(f"Error saving results to '{save_results_path}': {e}")
            raise # Re-raise saving error

    return results

# ----------------------------------------------------------
# Function to Load Saved Results (Unchanged)
# ----------------------------------------------------------
def load_bootstrap_results(filepath: str) -> Dict[str, Union[Dict, List[pd.DataFrame]]]:
    """
    Loads bootstrap analysis results from a pickle file.

    Args:
        filepath: Path to the saved pickle file.

    Returns:
        The loaded results dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError, pickle.UnpicklingError, ValueError: If loading or validation fails.
    """
    print(f"Attempting to load analysis results from: {filepath}")
    try:
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        print("Results loaded successfully.")
        # Basic validation
        if not isinstance(results, dict) or \
           'feature_types' not in results or \
           'category_counts' not in results or \
           'bootstrap_samples' not in results:
            raise ValueError("Loaded file does not contain expected results structure.")
        if not isinstance(results['bootstrap_samples'], list):
             raise ValueError("Loaded 'bootstrap_samples' is not a list.")
        if results['bootstrap_samples'] and not isinstance(results['bootstrap_samples'][0], pd.DataFrame):
             raise ValueError("Elements in loaded 'bootstrap_samples' are not DataFrames.")
        return results
    except FileNotFoundError:
        print(f"Error: Results file not found at {filepath}")
        raise
    except (IOError, pickle.UnpicklingError, ValueError) as e:
        print(f"Error loading or validating results from '{filepath}': {e}")
        raise