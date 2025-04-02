#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Pipes return the exit code of the last command to exit non-zero.
set -o pipefail

# --- Configuration ---
# Python package name in nixpkgs (must match the one used in flake.nix)
NIX_PYTHON_PKG="python311"
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

# Function to run poetry commands within a temporary nix shell
# that provides poetry itself and the correct python version.
run_poetry_command() {
    local poetry_args=("$@")
    info "Running: poetry ${poetry_args[*]}"
    # Use nix shell to provide poetry and the correct python version
    # Error handling: if nix shell fails, the script exits due to 'set -e'
    if ! nix shell "nixpkgs#$NIX_PYTHON_PKG" nixpkgs#poetry --command poetry "${poetry_args[@]}"; then
        # Although set -e should exit, add an explicit error message for clarity
        # This line might only be reached if set -e was somehow bypassed or if the command fails subtly.
        error "'poetry ${poetry_args[*]}' command execution failed within nix shell."
    fi
}
# --- End Helper Functions ---

# --- Main Script ---

# 1. Check Prerequisites
info "Checking prerequisites..."
if ! command -v nix &> /dev/null; then
    error "nix command not found. Please install Nix: https://nixos.org/download.html"
fi
if ! command -v git &> /dev/null; then
    error "git command not found. Please install Git."
fi
# Attempt to find project root relative to script location or current dir
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Check in script dir or current dir for pyproject.toml to find root
PROJECT_ROOT=""
if [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
    PROJECT_ROOT="$SCRIPT_DIR"
elif [ -f "./pyproject.toml" ]; then
    PROJECT_ROOT=$(pwd)
else
    # Try finding root using git if possible
    if command -v git &> /dev/null && git rev-parse --show-toplevel &> /dev/null; then
       PROJECT_ROOT=$(git rev-parse --show-toplevel)
    fi
fi

if [ -z "$PROJECT_ROOT" ] || [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
     error "pyproject.toml not found. Could not determine project root. Please run this script from the root of the OpenETLAgent project or ensure it's alongside pyproject.toml."
fi

# Change to project root for consistency
cd "$PROJECT_ROOT"
info "Running checks in project root: $PROJECT_ROOT"
info "Prerequisites found."

# 2. Check/Generate poetry.lock
info "Checking poetry.lock status..."
LOCK_NEEDED=false
if [ ! -f "poetry.lock" ]; then
    info "poetry.lock file not found."
    LOCK_NEEDED=true
else
    info "Checking poetry.lock consistency with pyproject.toml..."
    # Use run_poetry_command to check consistency
    # Hide success output (stdout & stderr) with >/dev/null 2>&1 for the check
    if ! nix shell "nixpkgs#$NIX_PYTHON_PKG" nixpkgs#poetry --command poetry check --lock > /dev/null 2>&1; then
        info "poetry.lock is outdated or inconsistent."
        LOCK_NEEDED=true
    else
        info "poetry.lock is up-to-date and consistent."
    fi
fi

# *** MODIFIED SECTION STARTS HERE ***
# Generate lock file if needed
if [ "$LOCK_NEEDED" = true ]; then
    info "Generating/updating poetry.lock (requires network access)..."
    # Always run 'poetry lock' if the lock file is missing or needs an update.
    # The run_poetry_command function handles calling it via nix shell.
    run_poetry_command lock
    # Error handling is inside run_poetry_command now. If it failed, script would exit.
    info "[✓] poetry.lock generated/updated successfully."
    warn "IMPORTANT: Please commit the updated poetry.lock file to Git:"
    warn "  git add poetry.lock"
    warn "  git commit -m \"Update poetry.lock\""
    sleep 2 # Give user a moment to read the warning
fi
# *** MODIFIED SECTION ENDS HERE ***

# 3. Check Git Status (Warn only)
info "Checking Git working tree status..."
if [[ -n $(git status --porcelain) ]]; then
    warn "Git working tree is dirty. For maximum reproducibility,"
    warn "consider committing changes before proceeding."
    sleep 2
else
    info "Git working tree is clean."
fi

# 4. Execute nix develop
info "Starting Nix development environment (nix develop)..."
info "This may take some time on the first run or if dependencies changed."

# Use 'exec' to replace the current script process with nix develop.
# This way, when the user exits the nix shell, they exit back to their original shell.
exec nix develop

# --- End Main Script ---