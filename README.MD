# Open ETL Agent

## Overview

This project provides a framework for defining and executing Extract, Transform, Load (ETL) pipelines. It processes data based on a central configuration file that defines inputs and outputs, and separate "flow" files that specify the transformation steps for each output.

A key feature is its ability to leverage Large Language Models (LLMs) like Gemini to automatically generate the sequence of transformation operations for a specific output if a corresponding flow file is missing. This generation happens step-by-step, allowing the LLM to build complex pipelines iteratively.

Key features include:

*   **Centralized Input/Output Definition:** Define all data sources and targets in a single `config.yaml` file.
*   **Modular Pipeline Flows:** Each output target has its own dedicated pipeline flow defined in a `pipelines/{output_key}.yaml` file, specifying the source input and the sequence of operations.
*   **Step-by-Step LLM Generation:** If a flow file for an output is missing, the system prompts an LLM iteratively to generate the required operations one by one, based on the source data and target schema. The generated flow is then saved for future use.
*   **Multiple Inputs/Outputs:** Support for multiple named input sources and multiple named output targets with different schemas and formats.
*   **Extensible Operations:** Includes common ETL operations (join, cast, arithmetic, comparison, fold, unfold, etc.) implemented using Polars.

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
    The script uses an LLM (e.g., Google Gemini) for operation generation when flow files are missing. You need to provide an API key.
    *   Create a file named `.env` in the project root.
    *   Add your API key to the `.env` file:
        ```dotenv
        GEMINI_API_KEY=your_api_key_here
        # Optional: Specify a different model (defaults to gemini-1.5-flash)
        # MODEL=gemini/gemini-1.5-pro-latest
        ```
    *   You can obtain a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Usage

Run the ETL process by providing the path to the central configuration YAML file:

```bash
python app/main.py path/to/your/config.yaml
```

The script performs the following steps:

1.  Loads the central configuration (`config.yaml`), which defines all available inputs and desired outputs.
2.  Iterates through each output defined in the configuration.
3.  For each output (`output_key`):
    *   Looks for a corresponding pipeline flow file at `pipelines/{output_key}.yaml`.
    *   **If the flow file exists:** Loads the `source` input key and the list of `operations` from the file.
    *   **If the flow file does *not* exist:** Initiates the step-by-step LLM generation process to create the sequence of operations needed to transform data from a suitable input (chosen by the LLM or defaulted) to match the target output schema. Saves the generated flow (source and operations) to `pipelines/{output_key}.yaml`.
    *   Loads the data from the specified `source` input.
    *   Applies the sequence of `operations` (either loaded or generated) to the data using Polars.
    *   Validates the transformed data against the schema defined for the current output.
    *   Saves the final transformed data to the path specified for the current output, using the defined format.

### Example

```bash
# Run the pipelines defined by config.yaml, generating missing flows
python app/main.py config.yaml
```

## Configuration Files

The ETL process relies on two types of YAML configuration files:

### 1. Central Configuration (`config.yaml`)

This file defines all the available input data sources and the desired output targets for the entire process.

```yaml
# Example config.yaml
inputs:
  # Logical name for the main input source
  orders_input:
    path: input_folder/orders.csv
    file_schema:
      name: RawOrders
      columns:
        order_id: { type: integer, description: Unique identifier for the order }
        customer_id: { type: integer, description: ID of the customer placing the order }
        order_date: { type: date, description: Date the order was placed (YYYY-MM-DD) }
        product_id: { type: integer, description: ID of the product ordered }
        quantity: { type: integer, description: Number of units ordered }
        # ... other raw order columns

  # Logical name for a lookup input source
  customers_input:
    path: input_folder/customers.csv
    file_schema:
      name: CustomerLookup
      columns:
        cust_id: { type: integer, description: Unique customer ID }
        customer_name: { type: string, description: Full name of the customer }
        region: { type: string, description: Customer's geographical region }
        # ... other customer details

outputs:
  # Logical name for the primary enriched output target
  enriched_orders_output:
    path: output_folder/enriched_orders_final.csv
    format: csv # csv, json, or parquet
    file_schema:
      name: EnrichedOrders
      columns:
        order_ref: { type: string, description: Prefixed order identifier (e.g., ORD-123) }
        customer_name: { type: string, description: Name of the customer }
        order_date: { type: date, description: Order date }
        product_id: { type: integer, description: Product ID }
        quantity: { type: integer, description: Quantity ordered }
        region: { type: string, description: Customer region }
        # ... other enriched columns

  # Logical name for a secondary, summarized output target
  summary_output:
    path: output_folder/order_summary.json
    format: json
    file_schema:
      name: OrderSummary
      columns:
        region: { type: string, description: Customer region }
        total_quantity: { type: integer, description: Total quantity ordered per region }
        # ... other summary columns
```

**Structure:**

*   **`inputs`**: A dictionary where each key is a unique logical name for an input source.
    *   `path`: Path to the input file (relative to the project root).
    *   `file_schema`: Defines the structure (`name`, `columns`) of this input file.
        *   `columns`: A dictionary where each key is the **column name**.
            *   `type`: Data type (`string`, `integer`, `float`, `boolean`, `date`, `positive integer`).
            *   `description`: Optional description for clarity and LLM guidance.
*   **`outputs`**: A dictionary where each key is a unique logical name for an output target (`output_key`).
    *   `path`: Path for the output file.
    *   `format`: Output file format (`csv`, `json`, `parquet`). Defaults to `csv`.
    *   `file_schema`: Defines the desired structure (`name`, `columns`) for this output file.

### 2. Pipeline Flow Files (`pipelines/{output_key}.yaml`)

These files define the specific steps (source and operations) required to generate one particular output defined in `config.yaml`. The filename must match the logical name of the output target (e.g., `pipelines/enriched_orders_output.yaml`). These files can be created manually or generated automatically by the LLM if they don't exist.

```yaml
# Example pipelines/enriched_orders_output.yaml
source: orders_input # Logical name of the input from config.yaml to start with
operations:
  - operation_type: assignation
    output_column: order_prefix
    value: "ORD-"
  - operation_type: casting
    input_column: order_id
    output_column: order_id_str
    target_type: string
  - operation_type: concatenation
    input_columns: [order_prefix, order_id_str]
    output_column: order_ref
    separator: ""
  - operation_type: bind # Join with customers data
    output_column: _ignored_ # Output column not strictly needed for bind
    right_file_path: customers_input # Use logical input name here
    right_schema_columns: { cust_id: integer, customer_name: string, region: string } # Schema of the right side
    left_on: customer_id # Column from the current data (orders_input)
    right_on: cust_id    # Column from the right data (customers_input)
    how: left
    columns_to_add: [customer_name, region] # Columns to bring from customers_input
  # ... more operations to match the EnrichedOrders schema ...
  - operation_type: equality # Example: Keep original order_date
    input_column: order_date
    output_column: order_date
  # ... potentially drop temporary columns like order_prefix, order_id_str
```

**Structure:**

*   **`source`**: The logical name (key from `config.yaml`'s `inputs` section) of the input data source to start this specific pipeline flow.
*   **`operations`**: A list defining the sequence of transformation operations to apply.

## Available Operations

The following operation types are currently supported:

*   **`equality`**: Copies an input column to an output column.
*   **`concatenation`**: Joins string values from multiple input columns into an output column, with an optional separator.
*   **`arithmetic`**: Performs basic math (`+`, `-`, `*`, `/`) between two input columns, storing the result in an output column.
*   **`comparison`**: Compares an input column to a literal value (`==`, `!=`, `>`, `<`, `>=`, `<=`), outputting a boolean result to the output column.
*   **`switching`**: Selects a value based on a condition. Uses a `switch_column` and a `mapping` dictionary. If the value in `switch_column` matches a key in `mapping`, the value from the column specified by the mapping's value is copied. A `default_column` or `default_value` handles non-matches.
*   **`casting`**: Changes the data type of an input column, saving to the output column. Supported types: `string`, `integer`, `float`, `boolean`, `date`, `positive integer`.
*   **`assignation`**: Creates an output column with a fixed literal value.
*   **`bind`**: Performs a join operation between the current DataFrame and another data source (specified by its logical input name or a direct file path). Requires defining the join keys (`left_on`, `right_on`), join type (`how`), the schema of the right-side data (`right_schema_columns`), and the columns to bring over (`columns_to_add`).
*   **`application`**: Applies a custom Python lambda function defined in `function_str` to rows. Uses `eval()`, exercise caution.
*   **`fold`**: Unpivots specified `value_columns` into two new columns: `key_column_name` (containing the original column names) and `value_column_name` (containing the values). `id_columns` are kept as identifiers.
*   **`unfold`**: Pivots data based on a `key_column` and `value_column`, using `index_columns` to group rows. Creates new columns based on unique values in the `key_column`.

Refer to `app/models.py` for detailed parameter definitions (Pydantic models) for each operation type.

## Command-Line Options

*   `config_file` (Positional Argument): Path to the central pipeline configuration YAML file (e.g., `config.yaml`). **Required**.

## Deactivating Environment

When you are finished, deactivate the environment:
```bash
conda deactivate
