#!/bin/bash

# This script will set up the Conda environment and install the dependencies

export PROJECT_NAME="llm_lwr_crag"
CONDA_ENV_NAME="${PROJECT_NAME}_env"
PYTHON_VERSION="3.11"

# Check if the conda environment exists
if conda env list | grep -q "$CONDA_ENV_NAME"; then
    echo "Conda environment '$CONDA_ENV_NAME' already exists. Updating it..."
    conda env update --name "$CONDA_ENV_NAME" --file environment.yml --prune
else
    echo "Conda environment '$CONDA_ENV_NAME' not found. Creating it from environment.yml..."
    conda env create --name "$CONDA_ENV_NAME" --file environment.yml
fi

# Activate the environment
conda activate "$CONDA_ENV_NAME"

# Install additional dependencies from requirements.txt if it exists
if [[ -f "requirements.txt" ]]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Skipping installation."
fi

echo "Environment setup complete."

# General environment variables
export PROJECT_ROOT="${PWD}/${PROJECT_NAME}/"
export DATA_DIR="${PROJECT_ROOT}/data/"
