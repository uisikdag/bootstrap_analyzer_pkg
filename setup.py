import os
import re
from setuptools import setup, find_packages

# Get the directory where setup.py resides - this makes paths more robust
setup_dir = os.path.abspath(os.path.dirname(__file__))

# Function to extract version from __init__.py
def get_version():
    version_file_path = os.path.join(setup_dir, "bootstrap_analyzer", "__init__.py")
    try:
        with open(version_file_path, "r", encoding="utf-8") as f:
            match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE)
    except FileNotFoundError:
         # Provide a more informative error if the file isn't where expected
         raise RuntimeError(f"Could not find version file at: {version_file_path}") from None

    if match:
        return match.group(1)
    raise RuntimeError(f"Version string not found in {version_file_path}")

# Function to read requirements
def get_requirements():
    req_file_path = os.path.join(setup_dir, "requirements.txt")
    try:
         with open(req_file_path, "r", encoding="utf-8") as f:
             # Filter out empty lines and comments
             return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
         print(f"Warning: requirements.txt not found at {req_file_path}. Installing without dependencies.")
         return []

# Function to read README for long description
def get_long_description():
    readme_file_path = os.path.join(setup_dir, "README.md")
    try:
         with open(readme_file_path, "r", encoding="utf-8") as f:
             return f.read()
    except FileNotFoundError:
         print(f"Warning: README.md not found at {readme_file_path}. Long description will be empty.")
         return ""

# --- Setup Configuration ---
setup(
    name='bootstrap_analyzer',
    version=get_version(),
    packages=find_packages(), # find_packages usually correctly finds 'bootstrap_analyzer'
    description='A package for analyzing features and generating bootstrap samples from data.',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    author='Your Name / AI Assistant', # Replace
    author_email='your.email@example.com', # Replace
    # url='https://github.com/yourusername/bootstrap_analyzer', # Optional: Replace
    install_requires=get_requirements(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License', # Or choose another license
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
    ],
    python_requires='>=3.7', # Specify compatible Python versions
)