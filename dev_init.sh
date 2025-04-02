#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Pipes return the exit code of the last command to exit non-zero.
set -o pipefail

# --- Configuration ---
CONDA_ENV_NAME="open-etl-agent"
ENV_FILE="environment.yml"
# --- End Configuration ---

# --- Helper Functions ---
info() {
    echo "[INFO] $@"
}

warn() {
    echo "[WARN] $@" >&2 # Write warnings to stderr
}

error() {
    echo "[ERROR] $@" >&2 # Write errors to stderr
    exit 1
}
# --- End Helper Functions ---

# --- Main Script ---

# 1. Check Prerequisites
info "Checking prerequisites..."
if ! command -v conda &> /dev/null; then
    error "conda command not found. Conda (or Miniconda/Miniforge/Mambaforge) is required. Please install it first. See installation instructions at: https://conda.io/projects/conda/en/latest/user-guide/install/index.html or https://github.com/conda-forge/miniforge#download"
fi
if [ ! -f "$ENV_FILE" ]; then
    error "Environment definition file '$ENV_FILE' not found in the current directory."
fi
info "Prerequisites found."

# 2. Create/Update Conda Environment
info "Checking/Updating Conda environment '$CONDA_ENV_NAME' from '$ENV_FILE'..."
info "This may take some time if packages need to be downloaded/installed."

# Use 'conda env update' which creates the environment if it doesn't exist,
# or updates it if it does. '--prune' removes packages not listed anymore.
if conda env update --name "$CONDA_ENV_NAME" --file "$ENV_FILE" --prune; then
    info "[✓] Conda environment '$CONDA_ENV_NAME' is up to date."
else
    # Catch potential errors during environment update/creation
    error "Failed to create or update Conda environment '$CONDA_ENV_NAME'."
fi

# 3. Activation Instruction
info "---------------------------------------------------------------------"
info "Conda environment '$CONDA_ENV_NAME' is ready."
info "To activate it, run:"
info "  conda activate $CONDA_ENV_NAME"
info "After activation, you can run the Python script."
info "To deactivate the environment later, run: conda deactivate"
info "---------------------------------------------------------------------"

# --- End Main Script ---
