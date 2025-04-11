# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 10:09:29 2025

@author: umit
"""

# bootstrap_analyzer/__init__.py

# Import key functions from modules to make them available at the package level
from .core import analyze_and_bootstrap_v3
from .io import run_bootstrap_analysis, load_bootstrap_results

# Define package version
__version__ = "0.1.0"

# Optional: Define what gets imported with 'from bootstrap_analyzer import *'
# Generally discouraged, but can be specified like this:
# __all__ = ['analyze_and_bootstrap_v3', 'run_bootstrap_analysis', 'load_bootstrap_results']