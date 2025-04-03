# Open ETL Agent

## Description
As the name indicate, the role of the project is to design an agent
being able to build a simple transformation script from an unknown format to a structured format.

This also include quality control of the data.

## Overview

This project provides a framework for defining and executing Extract, Transform, Load (ETL) pipelines using a declarative YAML configuration. It leverages Large Language Models (LLMs) like Gemini to automatically generate the sequence of transformation operations needed to convert data from defined input schemas to desired output schemas.

Key features include:

*   **Declarative Pipeline Definition:** Define inputs, outputs, schemas, and operations in a single YAML file.
*   **Multiple Inputs/Outputs:** Support for multiple named input sources and multiple named output targets with different schemas and formats.
*   **LLM-Powered Operation Generation:** Automatically generates transformation steps if not explicitly provided.
*   **Iterative Refinement:** If LLM-generated operations fail validation or execution, the system provides feedback to the LLM for correction attempts.
*   **Operation Flexibility:** Use operations embedded in the YAML, generate them via LLM, or load them from a separate file.
*   **Extensible Operations:** Includes common ETL operations (join, cast, arithmetic, comparison, etc.) implemented using Polars.

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

4.  **Set Environment Variables:**
    The script uses an LLM (currently Google Gemini) for operation generation. You need to provide an API key.
    *   Create a file named `.env` in the project root.
    *   Add your API key to the `.env` file:
        ```dotenv
        GEMINI_API_KEY=your_api_key_here
        ```
    *   You can obtain a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Usage

Run the ETL pipeline by providing the path to the pipeline definition YAML file:

```bash
python app/main.py path/to/your/pipeline_definition.yaml [OPTIONS]
```

The script will:

1.  Load the pipeline definition (inputs, outputs, schemas, operations).
2.  Load the primary input data (conventionally, the first input defined).
3.  Determine the operations to apply (see "Operation Sourcing" below).
4.  Apply the operations to the data using Polars.
5.  Validate the transformed data against the schema of each defined output.
6.  Save the transformed data to the specified output files in the defined formats.

### Example

```bash
# Run the pipeline defined in pipeline_definition.yaml
python app/main.py pipeline_definition.yaml

# Run the pipeline, but override operations with those from another file
python app/main.py pipeline_definition.yaml --use-operations generated_operations.yaml

# Run the pipeline and save the final used operations (whether generated or loaded)
python app/main.py pipeline_definition.yaml --save-operations final_ops.yaml
```

## Pipeline Definition YAML

The core of the process is the pipeline definition YAML file. It defines the structure and behaviour of the ETL process.

```yaml
# Example pipeline_definition.yaml
inputs:
  # Logical name for the main input
  orders:
    path: input_folder/orders.csv
    file_schema:
      name: RawOrders
      columns:
        order_id: { name: order_id, type: integer, description: ... }
        customer_id: { name: customer_id, type: integer, description: ... }
        # ... other columns
  # Logical name for a lookup input
  customers:
    path: input_folder/customers.csv
    file_schema:
      name: CustomerLookup
      columns:
        cust_id: { name: cust_id, type: integer, description: ... }
        customer_name: { name: customer_name, type: string, description: ... }
        # ... other columns

outputs:
  # Logical name for the primary output
  enriched_csv:
    path: output_folder/enriched_orders.csv
    format: csv # csv, json, or parquet
    file_schema:
      name: EnrichedOrders
      columns:
        order_ref: { name: order_ref, type: string, description: ... }
        customer_name: { name: customer_name, type: string, description: ... }
        # ... other columns
  # Logical name for a secondary, summarized output
  summary_json:
    path: output_folder/order_summary.json
    format: json
    file_schema:
      name: OrderSummary
      columns:
        order_ref: { name: order_ref, type: string, description: ... }
        total_price: { name: total_price, type: float, description: ... }

# Optional: Define operations directly
operations:
  - operation_type: assignation
    output_column: order_prefix
    value: "ORD-"
  - operation_type: casting
    # ... other operations
  - operation_type: bind
    output_column: ignored
    right_file_path: customers # Use logical input name here
    right_schema_columns: { cust_id: integer, customer_name: string, ... }
    left_on: customer_id
    right_on: cust_id
    how: left
    columns_to_add: [customer_name, customer_country]
  # ... more operations
```

**Structure:**

*   **`inputs`**: A dictionary where each key is a logical name for an input source.
    *   `path`: Path to the input file (relative to the project root).
    *   `file_schema`: Defines the structure (`name`, `columns`) of this input file.
        *   `columns`: A dictionary where each key is the column name.
            *   `name`: Column name (can be same as key).
            *   `type`: Data type (`string`, `integer`, `float`, `boolean`, `positive integer`).
            *   `description`: Optional description for clarity and LLM guidance.
*   **`outputs`**: A dictionary where each key is a logical name for an output target.
    *   `path`: Path for the output file.
    *   `format`: Output file format (`csv`, `json`, `parquet`). Defaults to `csv`.
    *   `file_schema`: Defines the desired structure (`name`, `columns`) for this output file.
*   **`operations`**: (Optional) A list defining the sequence of transformation operations. If omitted, the LLM will attempt to generate them.

## Operation Sourcing

The script determines which operations to apply based on the following priority:

1.  **`--use-operations <file>`**: If this command-line flag is provided, operations are loaded *exclusively* from the specified YAML file, overriding any operations defined within the main `pipeline_file`.
2.  **Embedded Operations**: If `--use-operations` is *not* used and the `pipeline_file` contains a non-empty `operations` list, those operations are used.
3.  **LLM Generation**: If neither `--use-operations` is used nor are operations embedded in the `pipeline_file`, the script will prompt the LLM to generate the operations based on the schemas of the first defined input and first defined output. It will attempt iterative refinement (up to `MAX_ATTEMPTS`) if the generated operations fail validation or execution.

## Available Operations

The following operation types are currently supported:

*   **`equality`**: Copies a column.
*   **`concatenation`**: Joins string values from multiple columns.
*   **`arithmetic`**: Performs basic math (`+`, `-`, `*`, `/`) between two columns.
*   **`comparison`**: Compares a column to a literal value (`==`, `!=`, `>`, `<`, `>=`, `<=`), outputs boolean.
*   **`switching`**: Selects a value from a column or literal based on a boolean condition column (`WHEN condition THEN true_value ELSE false_value`).
*   **`casting`**: Changes a column's data type.
*   **`assignation`**: Creates a column with a fixed literal value.
*   **`bind`**: Joins the current data with another input defined in the `inputs` section (using its logical name) or a direct file path.
*   **`application`**: Applies a custom Python lambda function (use with caution due to `eval()` usage).

Refer to `app/models.py` for detailed parameter definitions for each operation type.

## Command-Line Options

*   `pipeline_file` (Positional Argument): Path to the main pipeline definition YAML file. **Required**.
*   `--use-operations <path>` (Optional): Path to a YAML file containing an `operations` list. If provided, these operations will be used instead of any operations in `pipeline_file` or LLM generation.
*   `--save-operations <path>` (Optional): Path to save the final sequence of *successfully used* operations (whether loaded, embedded, or LLM-generated) as a YAML file. Useful for inspecting LLM output or creating reusable operation sets.

## Deactivating Environment

When you are finished, deactivate the environment:
```bash
    conda deactivate
    ```
