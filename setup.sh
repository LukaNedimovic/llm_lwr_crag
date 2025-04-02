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

echo "Environment setup complete."

# General environment variables
export PROJECT_ROOT=$PWD
export SRC_ROOT="${PROJECT_ROOT}/${PROJECT_NAME}/"
export DATA_DIR="${SRC_ROOT}/data/"
export DB_DIR="${SRC_ROOT}/db/"
export CONFIG_DIR="${SRC_ROOT}/config/"
export PERSIST_DIR="${SRC_ROOT}/persist/"
export PROMPTS_DIR="${SRC_ROOT}/prompts/"
export DOTENV_PATH="${PROJECT_ROOT}/.env"

export PYTHONPATH=$SRC_ROOT
