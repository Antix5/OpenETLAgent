import argparse
import yaml
import polars as pl
from polars import Utf8, Int64, Float64, Boolean # Explicit Polars types
from pathlib import Path
import sys
import logging
import warnings # To suppress Polars experimental warning for map_elements if needed
import os
import json
from dotenv import load_dotenv
import litellm
from typing import get_args, Tuple, List, Optional # Added Optional

# Assuming models.py is in the same directory or PYTHONPATH is set correctly
try:
    from models import (
        EtlPipeline, EqualityOperation, ConcatenationOperation,
        ApplicationOperation, SwitchingOperation, AssignationOperation,
        CastingOperation, ArithmeticOperation, ComparisonOperation, BindOperation, FileSchema # Added BindOperation
    )
except ImportError:
    # Simple fallback for running directly if app module isn't installed
    sys.path.insert(0, str(Path(__file__).parent.resolve()))
    from models import (
        EtlPipeline, EqualityOperation, ConcatenationOperation,
        ApplicationOperation, SwitchingOperation, AssignationOperation,
        CastingOperation, ArithmeticOperation, ComparisonOperation, BindOperation, FileSchema # Added BindOperation
    )

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s') # Changed level to DEBUG

# --- Type Mapping ---
# Map schema types to Polars types
POLARS_TYPE_MAP = {
    'string': Utf8,
    'integer': Int64,
    'float': Float64,
    'boolean': Boolean,
    'positive integer': Int64 # Validation handled by Pydantic, use Int64 here
    # Add more mappings as needed (e.g., Date, Datetime)
}

# --- LLM Tool Definitions ---

# Convert Pydantic models to JSON schema for LiteLLM tools
# We need descriptions for parameters to guide the LLM.
# We'll extract these from the Pydantic models or add them manually.

# Helper to get field descriptions or use field name
def get_description(model, field_name):
    field = model.model_fields.get(field_name)
    return field.description if field and field.description else f"The {field_name}"

# Define tools based on Pydantic models
LLM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "equality",
            "description": "Copies the value from an input column to an output column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_column": {"type": "string", "description": get_description(EqualityOperation, 'input_column')},
                    "output_column": {"type": "string", "description": get_description(EqualityOperation, 'output_column')},
                },
                "required": ["input_column", "output_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "concatenation",
            "description": "Concatenates values from multiple *existing* input columns into a single output column, optionally using a separator. To include a fixed prefix/suffix, first use 'assignation' to create a temporary column with the fixed value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_columns": {"type": "array", "items": {"type": "string"}, "description": "List of *existing* column names to concatenate."},
                    "separator": {"type": "string", "description": get_description(ConcatenationOperation, 'separator') + " (defaults to empty string)"},
                    "output_column": {"type": "string", "description": get_description(ConcatenationOperation, 'output_column')},
                },
                "required": ["input_columns", "output_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "application",
            "description": "Applies a custom Python lambda function using input columns to generate an output column. Use sparingly and carefully.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_columns": {"type": "array", "items": {"type": "string"}, "description": get_description(ApplicationOperation, 'input_columns')},
                    "function_str": {"type": "string", "description": get_description(ApplicationOperation, 'function_str')},
                    "output_column": {"type": "string", "description": get_description(ApplicationOperation, 'output_column')},
                },
                "required": ["input_columns", "function_str", "output_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switching",
            # Clarified description for column vs literal usage
            "description": "Selects a value based on a boolean condition column. Provide the condition column name. For true_column and false_column, provide either the name of the column to take the value from OR the literal string/value that should be outputted when the condition is true or false, respectively.",
            "parameters": {
                "type": "object",
                "properties": {
                    "condition_column": {"type": "string", "description": get_description(SwitchingOperation, 'condition_column') + " (must be boolean or castable to boolean)"},
                    # Clarified parameter descriptions
                    "true_column": {"type": "string", "description": "Column name or literal value to use when condition is true."},
                    "false_column": {"type": "string", "description": "Column name or literal value to use when condition is false."},
                    "output_column": {"type": "string", "description": get_description(SwitchingOperation, 'output_column')},
                },
                "required": ["condition_column", "true_column", "false_column", "output_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assignation",
            "description": "Assigns a fixed literal value to an output column.",
            "parameters": {
                "type": "object",
                "properties": {
                    # Changed type from "any" to "string" for Gemini compatibility.
                    # LLM should provide the value as a string (e.g., "123", "true", "some text").
                    # Added instruction to use schema value if specified.
                    "value": {"type": "string", "description": get_description(AssignationOperation, 'value') + " (provide as string, e.g., \"123\", \"true\", \"text\"). IMPORTANT: If the output schema description for this column specifies a fixed value, use that exact value here."},
                    "output_column": {"type": "string", "description": get_description(AssignationOperation, 'output_column')},
                },
                "required": ["value", "output_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "casting",
            "description": "Changes the data type of an input column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_column": {"type": "string", "description": get_description(CastingOperation, 'input_column')},
                    # Use get_args on the Literal annotation to get the allowed values
                    "target_type": {"type": "string", "enum": list(get_args(CastingOperation.model_fields['target_type'].annotation)), "description": get_description(CastingOperation, 'target_type')},
                    "output_column": {"type": "string", "description": get_description(CastingOperation, 'output_column')},
                },
                "required": ["input_column", "target_type", "output_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "arithmetic",
            "description": "Performs a basic arithmetic operation (+, -, *, /) between two input columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_columns": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2, "description": get_description(ArithmeticOperation, 'input_columns')},
                    "operator": {"type": "string", "enum": list(get_args(ArithmeticOperation.model_fields['operator'].annotation)), "description": get_description(ArithmeticOperation, 'operator')},
                    "output_column": {"type": "string", "description": get_description(ArithmeticOperation, 'output_column')},
                },
                "required": ["input_columns", "operator", "output_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "comparison",
            "description": "Compares an input column to a value using a specified operator (==, !=, >, <, >=, <=) and outputs a boolean result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_column": {"type": "string", "description": get_description(ComparisonOperation, 'input_column')},
                    "operator": {"type": "string", "enum": list(get_args(ComparisonOperation.model_fields['operator'].annotation)), "description": get_description(ComparisonOperation, 'operator')},
                    # Changed type from "any" to "string" for Gemini compatibility
                    "value": {"type": "string", "description": get_description(ComparisonOperation, 'value') + " (provide as string, e.g., \"3\", \"Shipped\", \"true\")"},
                    "output_column": {"type": "string", "description": get_description(ComparisonOperation, 'output_column')},
                },
                "required": ["input_column", "operator", "value", "output_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bind",
            "description": "Performs a join (default: left) between the current data and an external CSV file. Adds specified columns from the external file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "right_file_path": {"type": "string", "description": "The *relative path* from the project root to the CSV file to join with (e.g., 'input_folder/customers.csv')."},
                    "right_schema_columns": {"type": "object", "description": "Schema of the *right file* as a dictionary {column_name: type_string}. Include *only* the columns needed for the join key (`right_on`) and the `columns_to_add`."},
                    "left_on": {"type": "string", "description": "Column name from the *current* DataFrame to join on."},
                    "right_on": {"type": "string", "description": "Column name from the *right file* to join on (must exist in `right_schema_columns`)."},
                    "how": {"type": "string", "enum": list(get_args(BindOperation.model_fields['how'].annotation)), "description": "Type of join (default: 'left')."},
                    "columns_to_add": {"type": "array", "items": {"type": "string"}, "description": "List of *actual column names* from the *right file* to add to the result (must exist in `right_schema_columns`). Renaming happens in later steps if needed."},
                    # output_column is not needed for bind, but inherited. We can ignore it or make it optional in Pydantic if preferred.
                },
                "required": ["right_file_path", "right_schema_columns", "left_on", "right_on", "columns_to_add"],
            },
        },
    },
]

import re # Import re for cleaning LLM response

# --- LLM Operation Generation ---

def generate_operations_with_llm(
    input_schema: FileSchema,
    output_schema: FileSchema,
    previous_feedback: str | None = None # Added parameter for feedback
) -> list[dict]:
    """
    Generates the list of operations using an LLM with function calling.
    Optionally incorporates feedback from a previous failed attempt.
    """
    if previous_feedback:
        logging.info("Generating ETL operations using LLM with feedback from previous attempt...")
    else:
        logging.info("Generating ETL operations using LLM...")

    # Prepare schemas for the prompt
    input_schema_str = json.dumps(input_schema.model_dump(), indent=2)
    output_schema_str = json.dumps(output_schema.model_dump(), indent=2)

    prompt = f"""
You are an expert ETL pipeline generator. Your task is to determine the sequence of operations (function calls) needed to transform data from the input schema to the output schema.

Input Schema:
```json
{input_schema_str}
```

Output Schema:
```json
{output_schema_str}
```

Your task is to generate a YAML list representing the sequence of operations needed to transform the data from the input schema to the output schema.

Use the following operation types and adhere strictly to the specified YAML format for each:

1.  **equality**: Copies a column.
    ```yaml
    - operation_type: equality
      input_column: <source_column_name>
      output_column: <target_column_name>
    ```
2.  **concatenation**: Joins multiple columns.
    ```yaml
    - operation_type: concatenation
      input_columns:
        - <source_col_1>
        - <source_col_2>
      separator: "<separator_string>" # Optional, defaults to ""
      output_column: <target_column_name>
    ```
3.  **arithmetic**: Performs basic math (+, -, *, /).
    ```yaml
    - operation_type: arithmetic
      input_columns: [<col_a>, <col_b>] # Exactly two columns
      operator: <+, -, *, or />
      output_column: <target_column_name>
    ```
4.  **comparison**: Compares a column to a value, outputs boolean.
    ```yaml
    - operation_type: comparison
      input_column: <source_column_name>
      operator: <==, !=, >, <, >=, or <= >
      value: <value_to_compare_against> # Provide as string, e.g., "3", "Shipped"
      output_column: <target_column_name>
    ```
5.  **switching**: Selects a value based on a boolean condition.
    ```yaml
    - operation_type: switching
      condition_column: <boolean_column_name>
      true_column: <column_name_or_literal_value_if_true>
      false_column: <column_name_or_literal_value_if_false>
      output_column: <target_column_name>
    ```
    *Example*: To map `status_code == 3` to "Shipped" / "Not Shipped", you might first use `comparison` to create a boolean column (e.g., `is_shipped_bool`), then use `switching` with `condition_column: is_shipped_bool`, `true_column: "Shipped"`, `false_column: "Not Shipped"`.
6.  **casting**: Changes a column's data type.
    ```yaml
    - operation_type: casting
      input_column: <source_column_name>
      target_type: <string|integer|float|boolean>
      output_column: <target_column_name_or_same_as_input>
    ```
7.  **assignation**: Assigns a fixed literal value.
    ```yaml
    - operation_type: assignation
      value: "<the_literal_value>" # Value must be a string
      output_column: <target_column_name>
    ```
    *IMPORTANT*: If the output schema description specifies a fixed value (e.g., 'SystemA' for data_source), use that exact value.
    *Example*: To prefix `order_id` (already cast to string as `order_id_str`) with "ORD-", first use `assignation` to create `order_prefix` with value "ORD-", then use `concatenation` with `input_columns: [order_prefix, order_id_str]`.
8.  **bind**: Joins with another CSV file.
    ```yaml
    - operation_type: bind
      right_file_path: <path_to_other_csv>
      right_schema_columns: # Dictionary of column_name: type_string for the right file
        <right_col_1>: <type_string>
        <right_col_2>: <type_string>
        # ... include all columns needed for join key and columns_to_add
      left_on: <left_join_key_column>
      right_on: <right_join_key_column>
      how: <left|inner|outer|cross> # Default is 'left'
      columns_to_add: # List of columns to bring from the right file
        - <right_col_to_add_1>
        - <right_col_to_add_2>
      output_column: <target_column_name> # Note: This is inherited but ignored for bind
    ```
    *IMPORTANT*: Ensure `right_file_path` is the correct *relative path* from the project root (e.g., 'input_folder/customers.csv'). The `right_schema_columns`, `right_on`, and `columns_to_add` must use the *actual column names* from the file specified in `right_file_path`. Renaming to final output names happens later using `equality`.
9.  **application**: Applies a custom Python lambda (Use only if no other operation fits!).
    ```yaml
    - operation_type: application
      input_columns: [<col_a>, <col_b>, ...]
      function_str: "lambda r: <expression_using_r['col_a'], r['col_b']...>"
      output_column: <target_column_name>
    ```

Generate *only* the YAML list of operations required for the transformation, starting with `- operation_type: ...`. Do not include any introductory text, explanations, or markdown formatting like ```yaml ... ``` around the list.
"""
    # Add feedback to the prompt if provided
    if previous_feedback:
        feedback_prompt = f"""

IMPORTANT: The previous attempt to generate operations failed with the following feedback. Please analyze the feedback and generate a corrected sequence of operations that addresses these issues:
--- PREVIOUS FEEDBACK ---
{previous_feedback}
--- END PREVIOUS FEEDBACK ---

Please provide the corrected sequence of function calls:
"""
        prompt += feedback_prompt # Append feedback section

    try:
        # Ensure API key is loaded (redundant if load_dotenv() is called globally, but safe)
        if not os.getenv("GEMINI_API_KEY"):
             raise ValueError("GEMINI_API_KEY not found in environment variables.")

        logging.info("Calling LiteLLM with Gemini 2.0 Flash...") # Updated model name
        response = litellm.completion(
            model="gemini/gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}],
            # Removed tools and tool_choice
        )

        logging.info("LLM response received.")

        # Extract YAML content from the response message
        llm_output_content = None # Initialize for unbound error fix
        try:
            # Revert to standard nested access, suppress Pylance error
            llm_output_content = response.choices[0].message.content # type: ignore

            if not isinstance(llm_output_content, str):
                 logging.error(f"LLM response content is not a string or is missing: {llm_output_content}")
                 raise ValueError("LLM response content is not a string or is missing.")

            logging.debug(f"Raw LLM Output:\n{llm_output_content}")

            # Clean potential markdown fences ```yaml ... ```
            cleaned_yaml = re.sub(r'^```yaml\s*|\s*```$', '', llm_output_content, flags=re.MULTILINE).strip()

            # Parse the YAML content
            generated_operations = yaml.safe_load(cleaned_yaml)

            if not isinstance(generated_operations, list):
                 logging.error(f"LLM output was not a valid YAML list. Output:\n{cleaned_yaml}")
                 # Provide feedback for the next iteration
                 raise ValueError("LLM did not output a valid YAML list of operations.")

            logging.info(f"LLM generated {len(generated_operations)} operations (parsed from YAML).")
            return generated_operations

        except yaml.YAMLError as e:
            logging.error(f"Failed to parse YAML from LLM response: {e}\nRaw Output:\n{llm_output_content}", exc_info=True)
            # Provide feedback for the next iteration
            raise ValueError(f"LLM output was not valid YAML: {e}")
        except (AttributeError, IndexError, TypeError) as e:
             logging.error(f"Failed to extract content from LLM response structure: {e}", exc_info=True)
             raise ValueError(f"Could not extract content from LLM response: {e}")
        except Exception as e:
             logging.error(f"An unexpected error occurred processing LLM response: {e}", exc_info=True)
             raise # Re-raise other unexpected errors

    except Exception as e:
        logging.error(f"Error during LLM operation generation: {e}", exc_info=True)
        raise # Re-raise the exception


# --- Output Schema Validation ---

# Map Polars types back to schema string types for comparison
SCHEMA_TYPE_MAP_REVERSE = {v: k for k, v in POLARS_TYPE_MAP.items()}

def validate_output_schema(df: pl.DataFrame, target_schema: FileSchema) -> str | None:
    """
    Validates the DataFrame schema against the target output schema.

    Returns:
        None if the schema is valid.
        A string describing the discrepancies if invalid.
    """
    logging.info("Validating output DataFrame schema against target schema...")
    feedback_lines = []
    actual_schema = df.schema
    target_columns = target_schema.columns

    actual_col_names = set(actual_schema.keys())
    target_col_names = set(target_columns.keys())

    # Check for missing columns
    missing_cols = target_col_names - actual_col_names
    if missing_cols:
        feedback_lines.append(f"Missing columns required by output schema: {', '.join(sorted(list(missing_cols)))}")

    # # Check for extra columns - REMOVED: We will ignore extra columns here,
    # # as save_output_data selects only the required ones later.
    # extra_cols = actual_col_names - target_col_names
    # if extra_cols:
    #     # Provide this info in feedback but don't fail validation solely based on it?
    #     # For now, let's just ignore them in validation.
    #     logging.debug(f"Extra columns found but ignored by validation: {', '.join(sorted(list(extra_cols)))}")
    #     # feedback_lines.append(f"Extra columns found in output (should not be present): {', '.join(sorted(list(extra_cols)))}")

    # Check types only for columns required by the target schema
    common_cols = target_col_names.intersection(actual_col_names) # Base check on target cols that are present
    type_mismatches = []
    for col_name in common_cols:
        actual_type = actual_schema[col_name]
        target_type_str = target_columns[col_name].type
        target_pl_type = POLARS_TYPE_MAP.get(target_type_str)

        # Convert actual Polars type back to schema string type for comparison
        actual_type_str = SCHEMA_TYPE_MAP_REVERSE.get(actual_type)

        if not target_pl_type:
            logging.warning(f"Schema validation: Unknown target type '{target_type_str}' for column '{col_name}'. Skipping type check.")
            continue

        # Compare Polars types directly if possible, otherwise compare mapped strings
        if actual_type != target_pl_type:
             # Log the specific mismatch found
             logging.warning(f"Schema validation: Type mismatch for column '{col_name}'. Expected Polars type: {target_pl_type} (Schema: '{target_type_str}'), Actual Polars type: {actual_type} (Schema: '{actual_type_str}')")
             type_mismatches.append(f"Column '{col_name}': Expected type '{target_type_str}', Actual type: '{actual_type_str or actual_type}'") # Show actual Polars type if reverse mapping fails

    if type_mismatches:
        feedback_lines.append(f"Type mismatches: {'; '.join(type_mismatches)}")

    if feedback_lines:
        feedback = "Output schema validation failed:\n- " + "\n- ".join(feedback_lines)
        logging.warning(feedback)
        return feedback
    else:
        logging.info("Output schema validation successful.")
        return None


def load_pipeline_definition(file_path: Path) -> Tuple[FileSchema, FileSchema]:
    """Loads and validates the input and output schemas from the pipeline definition YAML."""
    logging.info(f"Loading pipeline definition schemas from: {file_path}")
    if not file_path.exists():
        logging.error(f"Pipeline definition file not found: {file_path}")
        raise FileNotFoundError(f"Pipeline definition file not found: {file_path}")
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        # Temporarily allow missing 'operations' for schema loading
        # We'll validate the generated operations later
        temp_data = data.copy()
        temp_data.pop('operations', None) # Remove operations if present
        # Validate schemas using a temporary dict or a modified model if needed
        # For simplicity, we assume EtlPipeline can be validated without 'operations'
        # or handle the validation error appropriately.
        # A cleaner way might be to have separate models for schema-only vs full pipeline.
        try:
             # Validate the structure minus operations
             # This might require adjusting EtlPipeline model or handling validation error
             # Let's assume for now Pydantic handles missing 'operations' if not required,
             # or we adjust the model later if needed.
             # A simple approach: create a temporary model or just extract schemas
             schemas_only = {"input_schema": data['input_schema'], "output_schema": data['output_schema']}
             # We don't strictly need the EtlPipeline object here anymore, just the schemas
             input_schema_obj = FileSchema.model_validate(data['input_schema'])
             output_schema_obj = FileSchema.model_validate(data['output_schema'])
             logging.info("Input and output schemas loaded and validated successfully.")
             # Return schemas instead of the full pipeline object initially
             return input_schema_obj, output_schema_obj
        except KeyError as e:
             logging.error(f"Missing required schema key in YAML: {e}")
             raise
        except Exception as e: # Catch Pydantic validation errors etc.
             logging.error(f"Error validating pipeline schemas: {e}")
             raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e: # Catch Pydantic validation errors etc.
        logging.error(f"Error validating pipeline definition: {e}")
        raise


def load_input_data(file_path: Path, schema: FileSchema) -> pl.DataFrame:
    """Loads input data based on the input schema using Polars."""
    logging.info(f"Loading input data from: {file_path}")
    if not file_path.exists():
        logging.error(f"Input data file not found: {file_path}")
        raise FileNotFoundError(f"Input data file not found: {file_path}")

    # TODO: Add support for other input formats based on schema.input_format
    if file_path.suffix.lower() != '.csv':
        raise NotImplementedError("Currently only CSV input format is supported.")

    # Prepare dtype mapping for polars read_csv
    dtype_map = {}
    schema_cols_dict = schema.columns
    for col_name, col_def in schema_cols_dict.items():
        pl_type = POLARS_TYPE_MAP.get(col_def.type)
        if pl_type:
            dtype_map[col_name] = pl_type
        else:
            logging.warning(f"No Polars type mapping found for schema type: {col_def.type} for column '{col_name}'. Polars will infer.")

    try:
        # Use infer_schema_length=0 to rely solely on provided schema_overrides where possible
        # Renamed 'dtypes' to 'schema_overrides'
        df = pl.read_csv(file_path, schema_overrides=dtype_map, infer_schema_length=1000 if not dtype_map else 0)
        logging.info(f"Input data loaded successfully. Shape: {df.shape}")

        # Basic validation: Check if all columns defined in schema exist
        schema_cols = set(schema_cols_dict.keys())
        df_cols = set(df.columns)
        missing_cols = schema_cols - df_cols
        if missing_cols:
            # Polars read_csv might skip columns not in dtypes if specified. Check if they exist at all.
            # This check might be less critical if dtypes are strictly enforced.
            logging.warning(f"Columns defined in input schema but potentially missing in CSV: {missing_cols}")
        extra_cols = df_cols - schema_cols
        if extra_cols:
             logging.info(f"Columns present in CSV but not defined in input schema: {extra_cols}")

        return df
    except Exception as e:
        logging.error(f"Error reading CSV file with Polars: {e}")
        raise


def apply_operations(df: pl.DataFrame, operations: list) -> pl.DataFrame:
    """Applies the sequence of ETL operations to the Polars DataFrame."""
    logging.info(f"Applying {len(operations)} operations using Polars...")
    temp_cols = set() # Keep track of temporary columns like _active_literal
    current_df = df # Work on a copy or new variable

    for i, op in enumerate(operations):
        logging.info(f"Applying operation {i+1}/{len(operations)}: {op.operation_type} -> {op.output_column}")
        try:
            if isinstance(op, EqualityOperation):
                if op.input_column not in current_df.columns:
                    raise ValueError(f"EqualityOperation: Input column '{op.input_column}' not found.")
                current_df = current_df.with_columns(
                    pl.col(op.input_column).alias(op.output_column)
                )

            elif isinstance(op, ConcatenationOperation):
                missing_inputs = [col for col in op.input_columns if col not in current_df.columns]
                if missing_inputs:
                    raise ValueError(f"ConcatenationOperation: Input columns not found: {missing_inputs}")
                # Ensure all input columns are cast to Utf8 before concatenation
                concat_expr = pl.concat_str(
                    [pl.col(c).cast(Utf8) for c in op.input_columns],
                    separator=op.separator
                ).alias(op.output_column)
                current_df = current_df.with_columns(concat_expr)

            elif isinstance(op, ApplicationOperation):
                missing_inputs = [col for col in op.input_columns if col not in current_df.columns]
                if missing_inputs:
                    raise ValueError(f"ApplicationOperation: Input columns not found: {missing_inputs}")

                # SECURITY WARNING: eval() is dangerous.
                # Using map_elements for arbitrary Python functions. This is slower than native expressions.
                logging.warning(f"Executing ApplicationOperation '{op.output_column}' using potentially slow map_elements and eval(). Ensure 'function_str' is trusted.")
                try:
                    # Compile the lambda string, providing a safe subset of builtins
                    safe_builtins = {
                        "float": float,
                        "int": int,
                        "str": str,
                        "list": list,
                        "dict": dict,
                        "set": set,
                        "tuple": tuple,
                        "True": True,
                        "False": False,
                        "None": None,
                        # Add other safe builtins if needed
                    }
                    lambda_func = eval(op.function_str, {"__builtins__": safe_builtins}, {}) # Pass safe builtins

                    # Define the Polars expression using map_elements
                    # Note: map_elements passes elements one by one, not the whole row dict like pandas apply
                    # We need to structure the lambda to accept arguments directly.
                    # Example: if function_str is "lambda v1, v2: v1 + v2" for input_columns ['Value1', 'Value2']
                    # This requires the lambda signature to match the number of input columns.
                    # A more robust approach might involve inspecting the lambda signature or requiring a specific format.
                    # For now, assume the lambda takes individual args.

                    # Let's pass the struct directly, as it behaves like a dict, matching the lambda in the YAML
                    # Defaulting return_dtype to Utf8 (string) as it's the most flexible for unknown lambda outputs.
                    # Downstream casting can be used if a different type is required.
                    map_expr = pl.struct(op.input_columns).map_elements(
                        lambda row_struct: lambda_func(row_struct), # Pass the struct directly
                        return_dtype=pl.Utf8
                    ).alias(op.output_column)

                    # Apply the map_elements expression
                    # Removed the warning suppression block as PolarsExperimentalWarning doesn't exist in this version
                    current_df = current_df.with_columns(map_expr)

                except Exception as e:
                    raise ValueError(f"Error executing function_str '{op.function_str}' for ApplicationOperation: {e}")


            elif isinstance(op, AssignationOperation):
                # Polars infers literal type, or we can cast if needed based on schema
                current_df = current_df.with_columns(
                    pl.lit(op.value).alias(op.output_column)
                )
                # Check if it's a temporary column for switching
                if op.output_column.startswith('_') and op.output_column.endswith('_literal'):
                    temp_cols.add(op.output_column)


            elif isinstance(op, SwitchingOperation):
                if op.condition_column not in current_df.columns:
                    raise ValueError(f"SwitchingOperation: Condition column '{op.condition_column}' not found.")

                # Reverted diagnostic step. Determine if true/false values are columns or literals.
                if op.true_column in current_df.columns:
                    true_val_expr = pl.col(op.true_column)
                    logging.debug(f"SwitchingOperation: Using column '{op.true_column}' for true value.")
                else:
                    true_val_expr = pl.lit(op.true_column)
                    logging.debug(f"SwitchingOperation: Using literal '{op.true_column}' for true value.")

                if op.false_column in current_df.columns:
                    false_val_expr = pl.col(op.false_column)
                    logging.debug(f"SwitchingOperation: Using column '{op.false_column}' for false value.")
                else:
                    false_val_expr = pl.lit(op.false_column)
                    logging.debug(f"SwitchingOperation: Using literal '{op.false_column}' for false value.")

                # Ensure condition column is boolean (Polars is stricter)
                condition_col_expr = pl.col(op.condition_column) # Define expression for condition column
                # Check the actual column's dtype before attempting cast in expression
                if current_df[op.condition_column].dtype != Boolean: # Use .dtype for direct access
                     logging.warning(f"SwitchingOperation: Condition column '{op.condition_column}' is not boolean type ({current_df[op.condition_column].dtype}). Attempting cast.")
                     try:
                         # Cast the expression, do not modify DataFrame mid-operation
                         condition_col_expr = pl.col(op.condition_column).cast(Boolean)
                     except Exception as e:
                         # Make the error message more specific about the casting failure
                         raise ValueError(f"Failed to cast condition column '{op.condition_column}' (type: {current_df[op.condition_column].dtype}) to boolean: {e}")

                # Construct the when/then/otherwise expression using prepared expressions
                switch_expr = pl.when(condition_col_expr) \
                                .then(true_val_expr) \
                                .otherwise(false_val_expr) \
                                .alias(op.output_column)
                current_df = current_df.with_columns(switch_expr)

            elif isinstance(op, CastingOperation):
                if op.input_column not in current_df.columns:
                    raise ValueError(f"CastingOperation: Input column '{op.input_column}' not found.")
                target_pl_type = POLARS_TYPE_MAP.get(op.target_type)
                if not target_pl_type:
                    raise ValueError(f"CastingOperation: Unsupported target type '{op.target_type}'")
                try:
                    current_df = current_df.with_columns(
                        pl.col(op.input_column).cast(target_pl_type).alias(op.output_column)
                    )
                except Exception as e:
                    # Provide more context in Polars errors
                    raise ValueError(f"Failed to cast column '{op.input_column}' to type '{op.target_type}' (Polars type: {target_pl_type}): {e}")

            elif isinstance(op, ArithmeticOperation):
                if len(op.input_columns) != 2:
                     raise ValueError(f"ArithmeticOperation requires exactly two input columns, got {len(op.input_columns)}.")
                col_a_name, col_b_name = op.input_columns
                if col_a_name not in current_df.columns:
                     raise ValueError(f"ArithmeticOperation: Input column '{col_a_name}' not found.")
                if col_b_name not in current_df.columns:
                     raise ValueError(f"ArithmeticOperation: Input column '{col_b_name}' not found.")

                col_a = pl.col(col_a_name)
                col_b = pl.col(col_b_name)

                if op.operator == '+':
                    arithmetic_expr = (col_a + col_b).alias(op.output_column)
                elif op.operator == '-':
                    arithmetic_expr = (col_a - col_b).alias(op.output_column)
                elif op.operator == '*':
                    arithmetic_expr = (col_a * col_b).alias(op.output_column)
                elif op.operator == '/':
                    # Consider adding division by zero handling if necessary
                    arithmetic_expr = (col_a / col_b).alias(op.output_column)
                else:
                    # Should be caught by Pydantic validation, but good to have defense
                    raise ValueError(f"ArithmeticOperation: Unsupported operator '{op.operator}'")

                current_df = current_df.with_columns(arithmetic_expr)

            elif isinstance(op, ComparisonOperation):
                if op.input_column not in current_df.columns:
                     raise ValueError(f"ComparisonOperation: Input column '{op.input_column}' not found.")

                # LLM provides value as string via tool definition. Create string literal.
                comparison_value_lit = pl.lit(op.value)
                # Cast the *input column* to string for comparison
                input_col_expr = pl.col(op.input_column).cast(pl.Utf8)
                logging.debug(f"ComparisonOperation: Comparing column '{op.input_column}' (cast to Utf8) with string literal '{op.value}'")


                # Apply the comparison operator using the column expression cast to string and the string literal
                if op.operator == '==':
                    comparison_expr = (input_col_expr == comparison_value_lit)
                elif op.operator == '!=':
                    comparison_expr = (input_col_expr != comparison_value_lit)
                elif op.operator == '>':
                    comparison_expr = (input_col_expr > comparison_value_lit)
                elif op.operator == '<':
                    comparison_expr = (input_col_expr < comparison_value_lit)
                elif op.operator == '>=':
                    comparison_expr = (input_col_expr >= comparison_value_lit)
                elif op.operator == '<=':
                    comparison_expr = (input_col_expr <= comparison_value_lit)
                else:
                    # Should be caught by Pydantic validation
                    raise ValueError(f"ComparisonOperation: Unsupported operator '{op.operator}'")

                current_df = current_df.with_columns(comparison_expr.alias(op.output_column))

            elif isinstance(op, BindOperation):
                logging.info(f"Binding with {op.right_file_path} on {op.left_on} = {op.right_on}")
                right_file_path = Path(op.right_file_path)
                if not right_file_path.exists():
                    raise FileNotFoundError(f"BindOperation: Right file not found at '{op.right_file_path}'")

                # Prepare dtypes for the right dataframe
                right_dtype_map = {}
                for col_name, type_str in op.right_schema_columns.items():
                    pl_type = POLARS_TYPE_MAP.get(type_str)
                    if pl_type:
                        right_dtype_map[col_name] = pl_type
                    else:
                        logging.warning(f"BindOperation: No Polars type mapping for right schema type '{type_str}' for column '{col_name}'. Polars will infer.")

                try:
                    df_right = pl.read_csv(right_file_path, schema_overrides=right_dtype_map, infer_schema_length=1000 if not right_dtype_map else 0)
                    logging.debug(f"Right DataFrame loaded. Shape: {df_right.shape}, Columns: {df_right.columns}")
                except Exception as e:
                    raise ValueError(f"BindOperation: Error reading right file '{op.right_file_path}': {e}") from e

                # Check if join keys exist
                if op.left_on not in current_df.columns:
                     raise ValueError(f"BindOperation: Left join key '{op.left_on}' not found in current DataFrame.")
                if op.right_on not in df_right.columns:
                     raise ValueError(f"BindOperation: Right join key '{op.right_on}' not found in right DataFrame ('{op.right_file_path}').")

                # Check if columns to add exist in the right dataframe
                missing_add_cols = [col for col in op.columns_to_add if col not in df_right.columns]
                if missing_add_cols:
                     raise ValueError(f"BindOperation: Columns specified in 'columns_to_add' not found in right DataFrame: {missing_add_cols}")

                # Perform the join
                try:
                    # Select only necessary columns from right df (join key + columns to add)
                    cols_to_select_from_right = list(set([op.right_on] + op.columns_to_add))
                    df_right_selected = df_right.select(cols_to_select_from_right)

                    current_df = current_df.join(
                        df_right_selected,
                        left_on=op.left_on,
                        right_on=op.right_on,
                        how=op.how
                    )
                    logging.debug(f"DataFrame shape after join: {current_df.shape}")
                except Exception as e:
                    raise ValueError(f"BindOperation: Error during join operation: {e}") from e

            else:
                logging.warning(f"Unsupported operation type encountered: {type(op)}. Skipping.")

        except Exception as e:
            logging.error(f"Error applying operation {i+1} ({op.operation_type} -> {op.output_column}): {e}")
            # Consider logging the state of current_df.head() here for debugging
            raise # Re-raise to stop processing

    # Clean up temporary columns
    if temp_cols:
        logging.info(f"Removing temporary columns: {temp_cols}")
        cols_to_drop = [col for col in temp_cols if col in current_df.columns]
        if cols_to_drop:
            current_df = current_df.drop(cols_to_drop)

    logging.info("All operations applied successfully.")
    return current_df


def save_output_data(df: pl.DataFrame, output_path: Path, schema: FileSchema):
    """Saves the transformed Polars DataFrame based on the output schema."""
    logging.info(f"Preparing output data for: {output_path}")

    output_format = schema.output_format or 'csv' # Default to csv
    output_columns_dict = schema.columns
    output_column_names = list(output_columns_dict.keys())

    # Select and reorder columns based on output schema
    final_df = df
    missing_output_cols = []
    present_output_cols = []

    if not output_column_names:
        logging.warning("No columns defined in output schema. Saving all columns.")
    else:
        logging.info(f"Selecting and ordering output columns: {output_column_names}")
        for col in output_column_names:
            if col in df.columns:
                present_output_cols.append(col)
            else:
                logging.warning(f"Column '{col}' defined in output schema but not found in final DataFrame. Skipping.")
                missing_output_cols.append(col)

        if not present_output_cols:
             logging.error("No columns specified in the output schema were found in the transformed data. Output file will be empty or invalid.")
             # Decide whether to raise error or write empty file
             # raise ValueError("No output columns found in the data.")
             final_df = df.select([]) # Create empty df with no columns
        else:
            final_df = df.select(present_output_cols) # Select only existing columns in the specified order

    logging.info(f"Saving output data to: {output_path} (Format: {output_format})")
    try:
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_format == 'csv':
            final_df.write_csv(output_path)
        elif output_format == 'json':
             # Polars write_json writes line-delimited JSON by default.
             # Use to_dicts() and standard json lib for pretty JSON array.
             import json
             with open(output_path, 'w') as f:
                 json.dump(final_df.to_dicts(), f, indent=2)
             # Or use write_ndjson for newline-delimited JSON
             # final_df.write_ndjson(output_path)
        elif output_format == 'parquet':
             # PyArrow is generally required for full Parquet support
             final_df.write_parquet(output_path)
        else:
            raise NotImplementedError(f"Output format '{output_format}' is not supported.")

        logging.info("Output data saved successfully.")
        if missing_output_cols:
             logging.warning(f"Note: Columns defined in output schema but missing from final data: {missing_output_cols}")

    except Exception as e:
        logging.error(f"Error saving output file: {e}")
        raise


MAX_ATTEMPTS = 3 # Maximum number of attempts for LLM refinement

def main():
    parser = argparse.ArgumentParser(description="OpenETLAgent: Apply ETL operations based on a YAML definition (using LLM for operations with iterative refinement) using Polars.")
    parser.add_argument("pipeline_file", type=Path, help="Path to the pipeline definition YAML file (containing schemas).")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Path to the input data file (e.g., CSV).")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Path for the output data file.")
    parser.add_argument("--save-operations", type=Path, default=None, help="Optional path to save the LLM-generated/used operations as a YAML file.")
    parser.add_argument("--use-operations", type=Path, default=None, help="Optional path to a YAML file containing pre-defined operations to use, skipping LLM generation.") # New argument

    args = parser.parse_args()

    try:
        # 1. Load Schemas
        input_schema, output_schema = load_pipeline_definition(args.pipeline_file)

        # 2. Load Input Data (do this once)
        df_initial = load_input_data(args.input, input_schema)

        final_operations = None
        df_transformed = None
        success = False

        # 3. Determine Operations: Load from file or Generate via LLM
        if args.use_operations:
            try:
                # Load and validate operations from the specified file
                validated_operations = load_operations_from_file(args.use_operations)
                final_operations = validated_operations # Use these operations directly

                # Apply the loaded operations
                logging.info(f"Applying {len(final_operations)} loaded operations...")
                df_transformed = apply_operations(df_initial.clone(), final_operations)
                logging.info("Loaded operations applied.")

                # Validate the output schema after applying loaded operations
                validation_feedback = validate_output_schema(df_transformed, output_schema)
                if validation_feedback is None:
                    logging.info("Output schema validated successfully for loaded operations.")
                    success = True
                else:
                    logging.error(f"Output schema validation failed for loaded operations: {validation_feedback}")
                    # Exit if loaded operations don't produce the correct schema
                    sys.exit(1)

            except Exception as e:
                 logging.error(f"Error processing --use-operations file '{args.use_operations}': {e}", exc_info=True)
                 sys.exit(1)
        else:
            # --- Iterative LLM Refinement Loop ---
            feedback = None
            validated_operations = [] # Initialize validated_operations before the loop

            for attempt in range(MAX_ATTEMPTS):
                logging.info(f"--- LLM Generation Attempt {attempt + 1}/{MAX_ATTEMPTS} ---")

                # Generate Operations using LLM (passing feedback from previous attempt)
                generated_ops_list = generate_operations_with_llm(input_schema, output_schema, previous_feedback=feedback)

                if not generated_ops_list:
                     logging.warning(f"Attempt {attempt + 1}: LLM did not generate any operations. Trying again if attempts remain.")
                     feedback = "The previous attempt generated no operations. Please generate a valid sequence."
                     continue # Skip to next attempt

                # Validate Generated Operations Syntactically (Pydantic)
                validated_operations = [] # Reset for current attempt
                logging.info("Validating LLM-generated operations syntax...")
                op_dict_for_validation = None # For error logging
                try:
                    dummy_pipeline_dict = {
                        "input_schema": input_schema.model_dump(),
                        "output_schema": output_schema.model_dump(),
                        "operations": []
                    }
                    for i, op_dict in enumerate(generated_ops_list):
                        op_dict_for_validation = op_dict # Store for potential error message
                        dummy_pipeline_dict["operations"] = [op_dict]
                        validated_pipeline = EtlPipeline.model_validate(dummy_pipeline_dict)
                        validated_operations.append(validated_pipeline.operations[0])
                    logging.info("LLM-generated operations syntax validated successfully.")
                except Exception as e:
                    logging.error(f"Attempt {attempt + 1}: Syntax validation failed for LLM-generated operation: {op_dict_for_validation}. Error: {e}", exc_info=True)
                    logging.error(f"Problematic operations list: {generated_ops_list}")
                    feedback = f"The generated operations failed syntax validation. Error on operation {op_dict_for_validation}: {e}. Please provide a syntactically correct list of operations."
                    continue # Skip to next attempt

                # Apply Operations
                try:
                     logging.info(f"Attempt {attempt + 1}: Applying {len(validated_operations)} validated operations...")
                     # Apply to a fresh copy of the initial data each time
                     df_transformed = apply_operations(df_initial.clone(), validated_operations)
                     logging.info(f"Attempt {attempt + 1}: Operations applied.")
                except Exception as e:
                     logging.error(f"Attempt {attempt + 1}: Error applying generated operations: {e}", exc_info=True)
                     feedback = f"Applying the generated operations resulted in an error: {e}. Please generate a sequence of operations that can be executed correctly on the input data."
                     continue # Skip to next attempt

                # Validate Output Schema
                feedback = validate_output_schema(df_transformed, output_schema)

                if feedback is None:
                    logging.info(f"Attempt {attempt + 1}: Output schema validated successfully!")
                    final_operations = validated_operations # Store the successful operations
                    success = True
                    break # Exit loop on success
                else:
                    logging.warning(f"Attempt {attempt + 1}: Output schema validation failed. Providing feedback to LLM.")
                    # Feedback is already set by validate_output_schema

            # --- End of LLM Loop ---

            if not success or df_transformed is None:
                 logging.error(f"Failed to generate a valid ETL pipeline after {MAX_ATTEMPTS} attempts.")
                 # Optionally save the last failed attempt's operations/output for debugging
                 if args.save_operations and validated_operations: # Use validated_operations from last attempt
                      logging.info(f"Saving last failed operations attempt to: {args.save_operations}")
                      ops_to_save = [op.model_dump() for op in validated_operations]
                      save_data = {'operations': ops_to_save, 'final_attempt_feedback': feedback}
                      try:
                          args.save_operations.parent.mkdir(parents=True, exist_ok=True)
                          with open(args.save_operations, 'w') as f:
                              yaml.dump(save_data, f, sort_keys=False, default_flow_style=False)
                      except Exception as e:
                          logging.error(f"Failed to save last failed operations: {e}")

                 sys.exit(1) # Exit with error code

        # --- Post Operations Processing ---

        # 4. Save Final Operations (either loaded or successfully generated)
        if args.save_operations and final_operations:
            logging.info(f"Saving final used operations to: {args.save_operations}")
            try:
                ops_to_save = [op.model_dump() for op in final_operations]
                save_data = {'operations': ops_to_save}
                args.save_operations.parent.mkdir(parents=True, exist_ok=True)
                with open(args.save_operations, 'w') as f:
                    yaml.dump(save_data, f, sort_keys=False, default_flow_style=False)
                logging.info("Final operations saved successfully.")
            except Exception as e:
                logging.error(f"Failed to save final operations: {e}", exc_info=True)
                # Don't fail the whole process if saving ops fails

        # 5. Save Final Output Data (already validated if success=True)
        if success and df_transformed is not None:
             save_output_data(df_transformed, args.output, output_schema)
             logging.info("ETL process completed successfully.")
        else:
             # This case should ideally be caught earlier, but as a safeguard:
             logging.error("ETL process failed to produce a valid final DataFrame.")
             sys.exit(1)

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        sys.exit(1)
    except NotImplementedError as e:
        logging.error(f"Feature not implemented: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during the ETL process: {e}", exc_info=True)
        sys.exit(1)


def load_operations_from_file(file_path: Path) -> List:
    """Loads and validates operations from a YAML file."""
    logging.info(f"Loading pre-defined operations from: {file_path}")
    if not file_path.exists():
        logging.error(f"Operations file not found: {file_path}")
        raise FileNotFoundError(f"Operations file not found: {file_path}")

    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)

        if 'operations' not in data or not isinstance(data['operations'], list):
            raise ValueError(f"YAML file '{file_path}' must contain a top-level 'operations' list.")

        ops_list = data['operations']
        validated_operations = []
        logging.info("Validating loaded operations syntax...")
        op_dict_for_validation = None # For error logging
        # Need input/output schema for full validation context, but can validate structure here
        dummy_pipeline_dict = {
            "input_schema": {"name": "DummyInput", "columns": {}}, # Minimal dummy schema
            "output_schema": {"name": "DummyOutput", "columns": {}}, # Minimal dummy schema
            "operations": []
        }
        for i, op_dict in enumerate(ops_list):
            op_dict_for_validation = op_dict
            dummy_pipeline_dict["operations"] = [op_dict] # Validate one by one
            try:
                validated_pipeline = EtlPipeline.model_validate(dummy_pipeline_dict)
                validated_operations.append(validated_pipeline.operations[0])
            except Exception as e:
                 logging.error(f"Syntax validation failed for loaded operation {i+1}: {op_dict_for_validation}. Error: {e}", exc_info=True)
                 raise ValueError(f"Syntax validation failed for loaded operation {i+1}: {e}")

        logging.info(f"Successfully loaded and validated {len(validated_operations)} operations.")
        return validated_operations

    except yaml.YAMLError as e:
        logging.error(f"Error parsing operations YAML file '{file_path}': {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading or validating operations from '{file_path}': {e}")
        raise


if __name__ == "__main__":
    main()
