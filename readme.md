# Stratified Bootstrap Sample Generator

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

1.  **Option A: git clone**
    ```
    git clone https://github.com/uisikdag/bootstrap_analyzer_pkg.git
    cd bootstrap_analyzer_pkg/examples
    python prep_and_test_ml.py
    ```

2.  **Option B: pip install**

    ```
    pip install git+https://github.com/uisikdag/bootstrap_analyzer_pkg.git
    ```
    download all files > bootstrap_analyzer_pkg/examples
     ```
    python prep_and_test_ml.py
    ```
