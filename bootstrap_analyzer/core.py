# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 10:08:49 2025

@author: umit
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional

# ----------------------------------------------------------
# Core Bootstrap Logic Function (Takes DataFrame as input)
# ----------------------------------------------------------
def analyze_and_bootstrap_v3(
    df: pd.DataFrame,
    x_cols: List[str],
    y_cols: List[str],
    n_samples: int,
    bootstrap_sample_size: Optional[int] = None,
    stratify_by: str = 'X', # Options: 'X', 'Y', 'both', 'none'
    categorical_threshold: int = 10,
    random_state: Optional[int] = None
) -> Dict[str, Union[Dict, List[pd.DataFrame]]]:
    """
    Analyzes DataFrame columns, identifies feature types, counts categories,
    and generates (optionally stratified) bootstrap samples of a specified size.

    Includes validation ensuring that if stratified sampling is used, the
    bootstrap_sample_size is not smaller than the smallest stratum size.

    Args:
        df: The input pandas DataFrame. **Must already be loaded.**
        x_cols: List of column names for independent variables (X).
        y_cols: List of column names for dependent variables (Y).
        n_samples: The number of bootstrap samples (M) to generate.
        bootstrap_sample_size: The desired size for each bootstrap sample.
                               If None, defaults to the size of the original
                               DataFrame. Must be between 1 and len(df).
                               If stratifying, must also be >= smallest stratum size.
                               Defaults to None.
        stratify_by: Specifies which variables to use for stratification.
                     Defaults to 'X'.
        categorical_threshold: Max unique numeric values for a column to be
                               considered categorical. Defaults to 10.
        random_state: Seed for the random number generator for reproducibility.
                      Defaults to None.

    Returns:
        A dictionary containing results: 'feature_types', 'category_counts',
        'bootstrap_samples'.

    Raises:
        ValueError: If validation checks fail (e.g., missing columns, invalid
                    sample size, stratification issues).
        TypeError: If input df is not a pandas DataFrame or other type issues.
    """

    # --- Input Validation ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty.")

    n_original = len(df)

    # Validate sample size basic range
    if bootstrap_sample_size is not None:
        if not isinstance(bootstrap_sample_size, int):
             raise TypeError("`bootstrap_sample_size` must be an integer or None.")
        if not (1 <= bootstrap_sample_size <= n_original):
            raise ValueError(f"`bootstrap_sample_size` must be between 1 and {n_original} (inclusive). Received: {bootstrap_sample_size}")
    effective_sample_size = bootstrap_sample_size if bootstrap_sample_size is not None else n_original

    # Validate column existence
    all_input_cols = x_cols + y_cols
    missing_cols = [col for col in all_input_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns specified in x_cols/y_cols not found in DataFrame: {missing_cols}")

    # Validate stratify_by option
    valid_stratify_options = ['X', 'Y', 'both', 'none']
    if stratify_by not in valid_stratify_options:
        raise ValueError(f"`stratify_by` must be one of {valid_stratify_options}")

    # Validate n_samples
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError(f"`n_samples` must be a positive integer. Received: {n_samples}")

    # --- Setup RNG ---
    base_rng = np.random.default_rng(random_state) if random_state is not None else np.random.default_rng()

    # --- 1. Detect Feature Types & 2. Count Categories ---
    feature_types: Dict[str, str] = {}
    category_counts: Dict[str, int] = {}
    all_df_cols = df.columns
    for col in all_df_cols:
        dtype = df[col].dtype
        unique_count = df[col].nunique(dropna=True)
        is_categorical = False
        if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
             if set(df[col].dropna().unique()) <= {True, False, 'True', 'False', 'true', 'false', 1, 0, '1', '0'}: is_categorical = True
             else: is_categorical = True
        elif pd.api.types.is_categorical_dtype(dtype): is_categorical = True
        elif pd.api.types.is_numeric_dtype(dtype):
            if unique_count <= categorical_threshold: is_categorical = True
            else: feature_types[col] = 'numerical'
        elif pd.api.types.is_bool_dtype(dtype): is_categorical = True
        else: feature_types[col] = 'other'

        if is_categorical:
            feature_types[col] = 'categorical'
            category_counts[col] = df[col].nunique(dropna=True)

    # --- 3. Determine Stratification Columns & Calculate Min Stratum Size ---
    stratification_cols: List[str] = []
    min_required_size = 1
    performing_stratification = False

    if stratify_by != 'none':
        cols_to_consider: List[str] = []
        if stratify_by == 'X': cols_to_consider = x_cols
        elif stratify_by == 'Y': cols_to_consider = y_cols
        elif stratify_by == 'both': cols_to_consider = list(set(x_cols + y_cols))

        stratification_cols = [
            col for col in cols_to_consider if feature_types.get(col) == 'categorical'
        ]

        if stratification_cols:
            performing_stratification = True
            try:
                stratum_sizes = df.groupby(stratification_cols, observed=True, dropna=False).size()
                if not stratum_sizes.empty:
                     min_required_size = stratum_sizes.min()
                     if min_required_size == 0 : min_required_size = 1
                else: min_required_size = 1
            except Exception as e:
                print(f"Warning: Error calculating stratum sizes: {e}. Minimum size constraint may not apply.")
                min_required_size = 1

    # --- 4. Final Validation of bootstrap_sample_size against Min Stratum Size ---
    if performing_stratification and bootstrap_sample_size is not None:
         if min_required_size > 0 and effective_sample_size < min_required_size:
             raise ValueError(
                 f"When stratifying by {stratification_cols}, the requested "
                 f"`bootstrap_sample_size` ({effective_sample_size}) must be at least "
                 f"the size of the smallest stratum ({min_required_size})."
             )

    # --- 5. Generate Bootstrap Samples ---
    bootstrap_samples: List[pd.DataFrame] = []
    original_indices = df.index

    for i in range(n_samples):
        current_seed = base_rng.integers(low=0, high=2**31)
        iter_rng = np.random.default_rng(current_seed)

        if performing_stratification:
            grouped = df.groupby(stratification_cols, observed=True, dropna=False)
            stratum_indices = [iter_rng.choice(g.index, size=len(g), replace=True)
                               for _, g in grouped if not g.empty]
            if not stratum_indices:
                 bootstrap_samples.append(pd.DataFrame(columns=df.columns))
                 continue
            full_stratified_indices = np.concatenate(stratum_indices)
            if len(full_stratified_indices) == 0:
                 bootstrap_samples.append(pd.DataFrame(columns=df.columns))
                 continue
            temp_stratified_df = df.loc[full_stratified_indices]

            if effective_sample_size < n_original:
                 n_to_sample = min(effective_sample_size, len(temp_stratified_df))
                 if n_to_sample > 0:
                     final_sample = temp_stratified_df.sample(n=n_to_sample, replace=False, random_state=iter_rng)
                 else: final_sample = pd.DataFrame(columns=df.columns)
            else: final_sample = temp_stratified_df
            bootstrap_samples.append(final_sample.reset_index(drop=True))

        else: # Simple bootstrap
            if n_original > 0 :
                 sampled_indices = iter_rng.choice(
                     original_indices, size=effective_sample_size, replace=True
                 )
                 bootstrap_samples.append(df.loc[sampled_indices].reset_index(drop=True))
            else:
                 bootstrap_samples.append(pd.DataFrame(columns=df.columns))

    # --- 6. Prepare Output ---
    results = {
        "feature_types": feature_types,
        "category_counts": category_counts,
        "bootstrap_samples": bootstrap_samples
    }
    return results