
from typing import Dict, Any, List, Optional
import logging
import json
import yaml
from app.models import FileDefinition, BaseOperation # Assuming these exist
import polars as pl # Assuming this is used

OPERATION_DEFINITIONS = """--- AVAILABLE OPERATIONS AND THEIR REQUIRED YAML SYNTAX ---

    You MUST use the following operation types and adhere strictly to their YAML syntax as defined by the provided Pydantic models.

    **IMPORTANT NOTES ON COLUMN REFERENCING AND TYPES:**
    - When specifying column names in operation fields like `input_columns`, `input_column`, `condition_column`, `true_column`, `false_column`, `left_on`, `right_on`, use the exact column name as it exists in the data *at that point in the operation sequence*. Refer to the Input/Output Schema definitions for available field names and their types.
    - **CRITICAL**: Pay close attention to data types defined in the schemas. Ensure operands in `arithmetic` and `comparison` operations are type-compatible before the operation runs. **YOU MUST insert `casting` operations where necessary *before* performing arithmetic or comparisons if types might mismatch** (e.g., comparing a 'string' column from the schema to a numeric `value`, or adding two columns where one is integer and one is float). The `value` field in `comparison` and `assignation` takes literals (strings, numbers, booleans).
    - **NOTE on bind output**: Columns added via the `bind` operation's `columns_to_add` list will keep their names from the right file's schema (e.g., adding `full_name` from customers adds a column named `full_name`). **Verify these names against the provided Input Schemas.** If the Target Output Schema requires different names for these bound columns (e.g., `customer_name`), **YOU MUST use an `equality` operation after the `bind`** to copy the value from the *correctly named* bound column (e.g., `full_name`) to the required output column name (e.g., `customer_name`).
    - **NOTE on concatenation prefix/suffix**: To add a fixed prefix or suffix using `concatenation`, first use `assignation` to create a column containing the literal prefix/suffix string, then use `concatenation` joining the prefix/suffix column and the data column(s).

    1.  **assignation**: Assigns a literal value to the output column.
        ```yaml
        - operation_type: assignation
          output_column: new_column_name # String: Name of column to create/overwrite
          value: literal_value # Literal (String, Number, Boolean)
        ```
        Syntax Example (Literal):

        ```yaml
        - operation_type: assignation
          output_column: column_with_As
          value: "A"
        ```

    2.  **equality**: Copies the `input_column` to the `output_column`. This is useful for renaming columns or duplicating them.
        ```yaml
        - operation_type: equality
          output_column: new_or_renamed_column # Target column name
          input_column: source_column # Column to copy from
        ```
        Syntax Example (Renaming after Bind):
        ```yaml
        # Assumes 'bind' previously added column 'full_name'
        - operation_type: equality
          output_column: customer_name # Target column name required by output schema
          input_column: full_name # Source column name added by bind
        ```

    3.  **concatenation**: Joins values from multiple input columns (`input_columns`) using an optional `separator`. See NOTE above for adding fixed prefixes/suffixes.
        ```yaml
        - operation_type: concatenation
          output_column: combined_output
          input_columns:
            - col_part1
            - col_part2
          separator: "-"
        ```
        Syntax Example (Joining two columns):
        ```yaml
        - operation_type: concatenation
          output_column: full_name
          input_columns:
            - first_name
            - last_name
          separator: " "
        ```
        Syntax Example (Adding prefix "UID-" to 'item_id'):
        ```yaml
        # Step 1: Assign the literal prefix
        - operation_type: assignation
          output_column: _prefix_ # Temporary column for prefix
          value: "ORD-"
        # Step 2: Concatenate prefix and id (no separator needed if joining directly)
        - operation_type: concatenation
          output_column: order_ref # Final output column
          input_columns:
            - _prefix_
            - order_id
          # separator: "" # Or omit if default is ""
        ```

    4.  **application**: Applies a Python lambda function (as a string) using specified input columns. The lambda acts on a row represented by 'r'. **NOTE**: This operation cannot reliably use built-in functions like `bool()`, `int()`, etc. Use `casting` for type conversions.
        **IMPORTANT**: The lambda function MUST be a string literal. Use `r['column_name']` to access column values within the lambda string, using the simple column name expected to be present at that stage.
        ```yaml
        - operation_type: application
          output_column: calculated_column
          input_columns:
            - col_a
            - col_b
          function_str: "lambda r: r['col_a'] + r['col_b']"
        ```
        Syntax Example:
        ```yaml
        - operation_type: application
          output_column: processed_data
          input_columns:
            - raw_data
          function_str: "lambda r: str(r['raw_data']).upper()"
        ```

    5.  **switching**: Copies a value from one of several source columns into the `output_column`, based on the value found in the `switch_column`. Requires a mapping of `switch_column` values to source column names, and a mandatory default action (copy from `default_column` or assign `default_value`) if no mapping key matches.
        ```yaml
        - operation_type: switching
          output_column: target_column # Name for the output column
          switch_column: column_to_check # Column whose value determines the source
          mapping: # Dictionary: {{value_in_switch_column: source_column_to_copy_from}}
            "ValueA": source_col_for_A
            "ValueB": source_col_for_B
            100: source_col_for_100 # Keys can be different types
          # --- Provide EXACTLY ONE default below ---
          default_column: default_source_col # OPTIONAL: Column to copy from if no mapping key matches
          default_value: "UNKNOWN" # OPTIONAL: Literal value to assign if no mapping key matches (and default_column is not used)
        ```
        Syntax Example:
        ```yaml
        # Assumes columns 'status', 'value_pending', 'value_completed', 'value_failed', 'value_default' exist
        - operation_type: switching
          output_column: final_value
          switch_column: status
          mapping:
            "PENDING": value_pending
            "COMPLETED": value_completed
            "FAILED": value_failed
          # Using default_column for any other status:
          default_column: value_default
        ```

    6.  **casting**: Changes the data type of the `input_column`. Outputs to `output_column`.
        ```yaml
        - operation_type: casting
          output_column: casted_column
          input_column: original_col
          target_type: integer
        ```
        Syntax Example:
        ```yaml
        - operation_type: casting
          output_column: amount_float
          input_column: amount_str
          target_type: float
        ```

    7.  **arithmetic**: Performs an arithmetic operation between exactly two `input_columns`. Outputs to `output_column`.
        ```yaml
        - operation_type: arithmetic
          output_column: calculation_result
          input_columns:
            - operand_col_1
            - operand_col_2
          operator: "*"
        ```

    8.  **comparison**: Compares `input_column` against a literal `value` using an `operator`. Outputs boolean to `output_column`.
        **CRITICAL**: Ensure the type of the `input_column` (check schema) is compatible with the literal `value` provided. If comparing a string column to a number, or vice-versa, **YOU MUST ADD A `casting` OPERATION BEFORE THIS COMPARISON** to make the types match.
        **NOTE**: This operation directly outputs a boolean result. If the target schema requires a boolean based on this comparison, simply use the target column name as the `output_column` here.
        ```yaml
        - operation_type: comparison
          output_column: comparison_flag
          input_column: col_to_compare
          operator: ">="
          value: 100
        ```
        Syntax Example 1 (Comparing numeric column to number):
        ```yaml
        - operation_type: comparison
          output_column: is_high_value
          input_column: amount
          operator: ">"
          value: 1000
        ```
        Syntax Example 2 (Comparing string column to string):
        ```yaml
        - operation_type: comparison
          output_column: is_status_pending
          input_column: status
          operator: "=="
          value: "PENDING"
        ```
        Syntax Example 3 (Comparing string column 'status_code' to number 3):
        ```yaml
        - operation_type: casting
          output_column: status_code_int
          input_column: status_code
          target_type: integer
        - operation_type: comparison
          output_column: is_shipped
          input_column: status_code_int
          operator: "=="
          value: 3
        ```

    9. **bind**: Joins the current data (left) with an external file (right) based on specified keys and adds selected columns from the right file. **Use input keys (e.g., 'customers_input') for `right_file_path`**. Ensure `right_on` and names in `columns_to_add` exactly match names in the right file's schema.
        ```yaml
        - operation_type: bind
          output_column: bind_placeholder
          right_file_path: "path/to/lookup_file.csv" # Original Example Path
          right_schema_columns: {{column_name: type_string}} # Define schema accurately here if not in Input Schemas
            key_column_in_right: string
            value_column_in_right: float
          left_on: join_key_in_left
          right_on: key_column_in_right # MUST exist in right schema
          how: "left"
          columns_to_add: # List[String]: Names MUST exist in right schema
            - value_column_in_right
        ```
        Syntax Example:
        ```yaml
        # Assumes customers file schema has 'cust_id', 'full_name', 'registration_country'
        - operation_type: bind
          output_column: ignored_bind_output
          right_file_path: "customers_input" # Use INPUT KEY here
          right_schema_columns: # Example if schema not provided elsewhere
            cust_id: integer
            full_name: string       # Correct Name
            registration_country: string # Correct Name
          left_on: customer_id
          right_on: cust_id           # Matches cust_id in right schema
          how: "left"
          columns_to_add:
            - full_name           # Add correct name
            - registration_country # Add correct name
        ```

    10. **fold**: Transforms data from wide to long format (like pandas melt). Gathers columns specified in `value_columns` into two new columns: one for the original column name (`key_column_name`) and one for the value (`value_column_name`). Columns listed in `id_columns` are kept as identifiers.
        ```yaml
        - operation_type: fold
          id_columns: # List[str]: Columns to keep as identifiers
            - id_col_1
            - id_col_2
          value_columns: # List[str]: Columns whose values will be folded
            - data_col_A
            - data_col_B
            - data_col_C
          key_column_name: new_key_col # str: Name for the new column holding 'data_col_A', 'data_col_B', etc.
          value_column_name: new_value_col # str: Name for the new column holding the corresponding values
        ```
        Syntax Example:
        ```yaml
        # Input has columns: product_id, region, sales_q1, sales_q2
        - operation_type: fold
          id_columns: [product_id, region]
          value_columns: [sales_q1, sales_q2]
          key_column_name: quarter
          value_column_name: sales
        # Output columns: product_id, region, quarter, sales
        # 'quarter' column will contain "sales_q1", "sales_q2"
        ```

    11. **unfold**: Transforms data from long to wide format (like pandas pivot). Spreads key-value pairs from `key_column` and `value_column` into new columns. `index_columns` identify the rows. **Note**: If the combination of `index_columns` and `key_column` is not unique, values will be aggregated using the 'first' value encountered. Pre-process data if different aggregation is needed.
        ```yaml
        - operation_type: unfold
          index_columns: # List[str]: Columns to use as row identifiers
            - id_col_1
            - id_col_2
          key_column: key_col # str: Column whose unique values become new column headers
          value_column: value_col # str: Column containing values for the new columns
        ```
        Syntax Example:
        ```yaml
        # Input has columns: product_id, region, quarter, sales (from previous fold example)
        - operation_type: unfold
          index_columns: [product_id, region]
          key_column: quarter
          value_column: sales
        # Output columns: product_id, region, sales_q1, sales_q2 (assuming 'quarter' had these values)
        ```

    --- END AVAILABLE OPERATIONS AND THEIR YAML SYNTAX ---
    """

def get_instruct_prompt(
    target_output_def: FileDefinition,
    inputs: Dict[str, FileDefinition]
) -> str:
    """
    Generates the formatted instruction prompt for the LLM based on pipeline details.
    Includes ALL input schemas and the TARGET output schema.
    Asks the LLM to determine the starting 'source' input key.
    """
    all_inputs_schema_str = ""
    input_keys = []
    if isinstance(inputs, dict):
        input_keys = list(inputs.keys())
        for input_key, input_def in inputs.items():
            if isinstance(input_def, FileDefinition) and hasattr(input_def, 'file_schema') and hasattr(input_def.file_schema, 'model_dump'):
                try:
                    schema_dump = input_def.file_schema.model_dump()
                    all_inputs_schema_str += f"Input Schema ({input_key}):\n"
                    all_inputs_schema_str += f"```json\n{json.dumps(schema_dump, indent=2)}\n```\n\n"
                except Exception as e:
                    logging.error(f"Error dumping schema for input '{input_key}': {e}")
                    all_inputs_schema_str += f"Input Schema ({input_key}): Error generating schema string.\n\n"
            else:
                 logging.warning(f"Input definition for key '{input_key}' is invalid or missing schema.")
                 all_inputs_schema_str += f"Input Schema ({input_key}): Invalid definition provided.\n\n"
    else:
         logging.error("Inputs definition is not a dictionary.")
         all_inputs_schema_str = "Error: Inputs definition is not a dictionary.\n\n"

    target_output_schema_str = "{ \"error\": \"Target output definition invalid\" }"
    target_output_schema_name = "ErrorSchema"
    target_output_path = "unknown"
    if isinstance(target_output_def, FileDefinition) and hasattr(target_output_def, 'file_schema') and hasattr(target_output_def.file_schema, 'model_dump'):
        try:
            target_output_schema_dump = target_output_def.file_schema.model_dump()
            target_output_schema_str = json.dumps(target_output_schema_dump, indent=2)
            target_output_schema_name = getattr(target_output_def.file_schema, 'name', 'UnnamedSchema')
            target_output_path = getattr(target_output_def, 'path', 'unknown')
        except Exception as e:
            logging.error(f"Error dumping schema for target output: {e}")
            target_output_schema_str = "{ \"error\": \"Error generating target output schema string.\" }"
    else:
         logging.error("Target output definition is invalid or missing schema.")

    prompt = f"""
    You are an expert ETL pipeline generator. Your task is to determine the sequence of operations AND the best starting input ('source') needed to transform data from one or more of the available Input Schemas into data matching the Target Output Schema.

    --- AVAILABLE INPUTS ---
    {all_inputs_schema_str}
    --- END AVAILABLE INPUTS ---

    --- TARGET OUTPUT ---
    Target Output Schema ({target_output_schema_name}):
    ```json
    {target_output_schema_str}
    ```
    Target Output Path: {target_output_path}
    --- END TARGET OUTPUT ---

    Available Input File Keys (use these names when referencing input data, e.g., in `bind`'s `right_file_path`): {input_keys}

    Your goal is to:
    1.  **Determine the most logical starting `source` input key** from the 'Available Input File Keys' list. This is the input the pipeline should load first.
    2.  **Generate the sequence of `operations`** needed to transform data starting from that chosen `source` into data matching the Target Output Schema, using ONLY the available operations defined below. You can use `bind` operations to bring in data from other available inputs as needed.

    {OPERATION_DEFINITIONS}

    Generate a YAML dictionary containing the chosen `source` key and the required `operations` list. The structure MUST be:
    ```yaml
    source: chosen_input_key_from_available_list
    operations:
      - operation_type: ...
        # ... parameters ...
      - operation_type: ...
        # ... parameters ...
      # ... etc ...
    ```

    Before answering, you can reason a bit but the answer has to follow the format and only one answer has to be in your output

    Use ONLY the operations listed above and strictly adhere to the specified YAML syntax. Base the sequence of operations SOLELY on the Input/Output Schema mapping provided above. **Pay extremely close attention to using correct column names from the provided schemas, especially for `bind` operations.**
    Remember to insert `casting` operations when types are incompatible for comparison or arithmetic. Ensure the final set of columns and their types match the Target Output Schema.

    """
    return prompt


def get_next_step_prompt(
    current_schema_dict: Dict[str, str],
    target_output_def: FileDefinition,
    inputs: Dict[str, FileDefinition],
    operations_history: List[Dict[str, Any]],
    current_state_header: str,
    previous_feedback: Optional[str] = None
) -> str:
    """
    Generates a prompt asking the LLM for the single next operation
    to apply, given the current state and target.
    """
    current_schema_str = json.dumps(current_schema_dict, indent=2)

    target_output_schema_str = "{ \"error\": \"Target output definition invalid\" }"
    target_output_schema_name = "ErrorSchema"
    if isinstance(target_output_def, FileDefinition) and hasattr(target_output_def, 'file_schema') and hasattr(target_output_def.file_schema, 'model_dump'):
        try:
            target_output_schema_dump = target_output_def.file_schema.model_dump()
            target_output_schema_str = json.dumps(target_output_schema_dump, indent=2)
            target_output_schema_name = getattr(target_output_def.file_schema, 'name', 'UnnamedSchema')
        except Exception as e:
            logging.error(f"Error dumping schema for target output: {e}")
            target_output_schema_str = "{ \"error\": \"Error generating target output schema string.\" }"
    else:
        logging.error("Target output definition is invalid or missing schema.")

    history_str = yaml.dump(operations_history, sort_keys=False, default_flow_style=False) if operations_history else "None"

    input_keys = list(inputs.keys()) if isinstance(inputs, dict) else []

    prompt = f"""
    You are an expert ETL pipeline generator working step-by-step. Your task is to determine the **single next operation** needed to incrementally transform data with the 'Current Schema' towards the 'Target Output Schema'.

    --- CURRENT STATE ---
    Current Schema:
    ```json
    {current_schema_str}
    ```

    the begining of the table looks like this :
    {current_state_header}

    Operations Applied So Far:
    ```yaml
    {history_str}
    ```
    --- END CURRENT STATE ---

    --- TARGET OUTPUT ---
    Target Output Schema ({target_output_schema_name}):
    ```json
    {target_output_schema_str}
    ```
    --- END TARGET OUTPUT ---

    --- CONTEXT ---
    Available Input File Keys (for use in `bind`'s `right_file_path`): {input_keys}
    --- END CONTEXT ---

    Your goal is to:
    1.  Analyze the 'Current Schema', 'Target Output Schema', and 'Operations Applied So Far'.
    2.  Determine the **single best next operation** from the 'Available Operations' list below that makes progress towards the target. Consider missing columns, type mismatches, or necessary calculations.
    3.  If the 'Current Schema' already matches the 'Target Output Schema' and no more operations are needed, output the special operation: `- operation_type: done`.

    {OPERATION_DEFINITIONS}

    **In addition to the operations above, you can output the following special operation:**

    12. **done**: Special operation type to indicate the pipeline is complete and the current schema matches the target.
        ```yaml
        - operation_type: done
        ```

    Generate *only* the YAML for the **single next operation** required for the transformation, starting *exactly* with `- operation_type: ...`.
    Use ONLY the operations listed above (including `done`) and strictly adhere to the specified YAML syntax. Base your choice on the 'Current Schema' and 'Target Output Schema'. **Pay extremely close attention to using correct column names.** Remember to insert `casting` operations when types are incompatible.
    If the pipeline is complete, output ONLY `operation_type: done`.
    
    You can think before asnwering but the output operation has to be within ```yaml ...```
    """

    if previous_feedback:
        feedback_prompt = f"""

        IMPORTANT: The previous step failed with the following feedback. Please analyze the feedback and generate a corrected single next operation:
        --- PREVIOUS FEEDBACK ---
        {previous_feedback}
        --- END PREVIOUS FEEDBACK ---

        Please provide the corrected single next operation YAML:
        """
        prompt += feedback_prompt
    else:
         if not operations_history:
              first_step_instruction = "\nThis is the first step. Choose the first operation to apply based on the Current Schema and Target Output Schema."
              prompt += first_step_instruction

    return prompt
