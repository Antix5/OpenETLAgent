
from typing import Dict, Any
import logging

def get_instruct_prompt(
    primary_input_key: str,
    input_schema_str: str,
    output_schema_str: str,
    primary_output_key: str,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any]
) -> str:
    """
    Generates the formatted instruction prompt for the LLM based on pipeline details.
    """
    prompt = f"""
You are an expert ETL pipeline generator. Your task is to determine the sequence of operations needed to transform data matching the Input Schema into data matching the Output Schema.

Input Schema ({primary_input_key}):
```json
{input_schema_str}
```

Output Schema ({primary_output_key}):
```json
{output_schema_str}
```

Available Input Files (use these names when referencing input data, e.g., in `bind`'s `right_file_path`): {list(inputs.keys())}
Available Output Files (pipeline destination): {list(outputs.keys())}

Your goal is to generate the sequence of operations below that correctly transforms data matching the Input Schema into data matching the Output Schema, using ONLY the available operations defined below.

--- AVAILABLE OPERATIONS AND THEIR REQUIRED YAML SYNTAX ---

You MUST use the following operation types and adhere strictly to their YAML syntax as defined by the provided Pydantic models.

**IMPORTANT NOTES ON COLUMN REFERENCING AND TYPES:**
- When specifying column names in operation fields like `input_columns`, `input_column`, `condition_column`, `true_column`, `false_column`, `left_on`, `right_on`, use the exact column name as it exists in the data *at that point in the operation sequence* (usually WITHOUT the 'table_name.' prefix like '{primary_input_key}.'). Refer to the Input/Output Schema definitions for available field names and their types.
- **CRITICAL**: Pay close attention to data types defined in the schemas. Ensure operands in `arithmetic` and `comparison` operations are type-compatible before the operation runs. **YOU MUST insert `casting` operations where necessary *before* performing arithmetic or comparisons if types might mismatch** (e.g., comparing a 'string' column from the schema to a numeric `value`, or adding two columns where one is integer and one is float). The `value` field in `comparison` and `assignation` takes literals (strings, numbers, booleans).
- **NOTE on bind output**: Columns added via the `bind` operation's `columns_to_add` list will keep their names from the right file's schema (e.g., adding `full_name` from customers adds a column named `full_name`). **Verify these names against the provided Input Schemas.** If the Output Schema requires different names for these bound columns (e.g., `customer_name`), **YOU MUST use an `assignation` operation after the `bind`** to copy the value from the *correctly named* bound column (e.g., `full_name`) to the required output column name (e.g., `customer_name`).
- **NOTE on concatenation prefix/suffix**: To add a fixed prefix or suffix using `concatenation`, first use `assignation` to create a column containing the literal prefix/suffix string, then use `concatenation` joining the prefix/suffix column and the data column(s).

1.  **assignation**: Assigns a value to the output column. The value can be a literal OR copied from another existing column.
    ```yaml
    - operation_type: assignation
      output_column: new_column_name # String: Name of column to create/overwrite
      # Provide EITHER a literal value OR the name of an existing column to copy data FROM
      value: source_column_or_literal # String (for source column name) OR Literal (String, Number, Boolean)
    ```
    Syntax Example (Literal):
    ```yaml
    - operation_type: assignation
      output_column: status_description
      value: "Processed"
    ```
    Syntax Example (From Column after Bind):
    ```yaml
    # Assumes 'bind' previously added columns 'full_name' and 'registration_country'
    - operation_type: assignation
      output_column: customer_name # Target column name required by output schema
      value: full_name # Source column name added by bind (MUST match actual name)
    - operation_type: assignation
      output_column: customer_country
      value: registration_country # Source column name added by bind (MUST match actual name)
    ```

2.  **equality**: Checks if values in `input_column` are equal (purpose unclear from model, requires implementation details or model update). Outputs boolean to `output_column`.
    ```yaml
    - operation_type: equality
      output_column: is_equal_flag
      input_column: column_to_check
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
    Syntax Example (Adding prefix "ORD-" to 'order_id'):
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

5.  **switching**: Selects value from `true_column` or `false_column` based on the boolean `condition_column`. Outputs to `output_column`.
    ```yaml
    - operation_type: switching
      output_column: result_column
      condition_column: boolean_flag_col
      true_column: value_if_true_col
      false_column: value_if_false_col
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

9. **bind**: Joins the current data (left) with an external file (right) based on specified keys and adds selected columns from the right file. **Ensure `right_on` and names in `columns_to_add` exactly match names in the right file's schema.**
    ```yaml
    - operation_type: bind
      output_column: bind_placeholder
      right_file_path: "path/to/lookup_file.csv"
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
      right_file_path: "lookup_file.csv"
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

--- END AVAILABLE OPERATIONS AND THEIR YAML SYNTAX ---

Generate *only* the YAML list of operations required for the transformation, starting *exactly* with `- operation_type: ...`.
Use ONLY the operations listed above and strictly adhere to the specified YAML syntax derived from the Pydantic models. Base the sequence of operations and their parameters SOLELY on the Input/Output Schema mapping provided above. **Pay extremely close attention to using correct column names from the provided schemas, especially for `bind` operations.** Remember to insert `casting` operations when types are incompatible for comparison or arithmetic. Ensure the final set of columns and their types match the Output Schema.
Do not include any introductory text, explanations, comments, or markdown formatting like ```yaml ... ``` around the final YAML list output.
"""
    return prompt
