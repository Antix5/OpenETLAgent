# Open ETL Agent

## Description
As the name indicate, the role of the project is to design an agent
being able to build a simple transformation script from an unknown format to a structured format.

This also include quality control of the data.

!!! THIS PROJECT JUST STARTED, NOT USABLE YET !!!

## Setup

This project uses Conda for environment management.

1.  **Prerequisite: Install Conda**
    If you don't have Conda installed, please install Miniconda or Miniforge first.
    *   Miniconda: [https://conda.io/projects/conda/en/latest/user-guide/install/index.html](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
    *   Miniforge/Mambaforge (recommended, uses conda-forge by default): [https://github.com/conda-forge/miniforge#download](https://github.com/conda-forge/miniforge#download)

2.  **Create/Update Environment:**
    Run the setup script. This will create (or update) a Conda environment named `open-etl-agent` using the `environment.yml` file.
    ```bash
    ./dev_init.sh
    ```

3.  **Activate Environment:**
    Activate the created environment in your terminal:
    ```bash
    conda activate open-etl-agent
    ```
    Your terminal prompt should now show `(open-etl-agent)` at the beginning.

4.  **Run the Example:**
    While the environment is active, you can run the example pipeline:
    ```bash
    python app/main.py pipeline_definition.yaml --input input_folder/sample_input.csv --output output_folder/sample_output.csv
    ```

5.  **Deactivate Environment:**
    When you are finished, deactivate the environment:
    ```bash
    conda deactivate
    ```
