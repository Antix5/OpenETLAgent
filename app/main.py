import argparse
import yaml
import polars as pl
from pathlib import Path
import sys
import logging
import os
import json
from dotenv import load_dotenv
import litellm
from typing import List, Optional, Dict, Sequence # Added Sequence
import re # Import re for cleaning LLM response

# Attempt standard import first
from app.models import (
    EtlPipeline, FileDefinition, BaseOperation, # Added BaseOperation
    POLARS_TYPE_MAP, SCHEMA_TYPE_MAP_REVERSE,
)
from app.instruct_prompt import get_instruct_prompt
from app.operations.equality import apply_equality
from app.operations.concatenation import apply_concatenation
from app.operations.application import apply_application
from app.operations.asignation import apply_assignation
from app.operations.switching import apply_switching
from app.operations.casting import apply_casting
from app.operations.arithmetic import apply_arithmetic
from app.operations.comparison import apply_comparison
from app.operations.bind import apply_bind

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Operation Dispatcher ---
# Map operation_type string to the corresponding function
OPERATION_DISPATCHER = {
    'equality': apply_equality,
    'concatenation': apply_concatenation,
    'application': apply_application,
    'assignation': apply_assignation,
    'switching': apply_switching,
    'casting': apply_casting,
    'arithmetic': apply_arithmetic,
    'comparison': apply_comparison,
    'bind': apply_bind,
}


# --- LLM Operation Generation ---

def generate_operations_with_llm(
    inputs: Dict[str, FileDefinition],
    outputs: Dict[str, FileDefinition],
    previous_feedback: str | None = None
) -> list[dict]:
    """
    Generates the list of operations using an LLM.
    Uses the schema of the *first* input and *first* output for context.
    Optionally incorporates feedback from a previous failed attempt.
    """
    if not inputs:
        raise ValueError("Cannot generate operations: No inputs defined in the pipeline.")
    if not outputs:
        raise ValueError("Cannot generate operations: No outputs defined in the pipeline.")

    # Use the first input and first output for schema context in the prompt
    # A more sophisticated approach might allow specifying primary input/output
    primary_input_key = list(inputs.keys())[0]
    primary_output_key = list(outputs.keys())[0]
    input_schema = inputs[primary_input_key].file_schema
    output_schema = outputs[primary_output_key].file_schema
    logging.info(f"Using schema from input '{primary_input_key}' and output '{primary_output_key}' for LLM generation.")


    if previous_feedback:
        logging.info("Generating ETL operations using LLM with feedback from previous attempt...")
    else:
        logging.info("Generating ETL operations using LLM...")

    input_schema_str = json.dumps(input_schema.model_dump(), indent=2)
    output_schema_str = json.dumps(output_schema.model_dump(), indent=2)

    # Simplified prompt structure (assuming LLM knows the operations from context/previous turns if needed)
    # Or re-include the full operation list if generation quality degrades.
    prompt = get_instruct_prompt(
        primary_input_key=primary_input_key,
        input_schema_str=input_schema_str,
        output_schema_str=output_schema_str,
        primary_output_key=primary_output_key,
        inputs=inputs,
        outputs=outputs
    )

    if previous_feedback:
        feedback_prompt = f"""

        IMPORTANT: The previous attempt failed with the following feedback. Please analyze the feedback and generate a corrected sequence of operations:
        --- PREVIOUS FEEDBACK ---
        {previous_feedback}
        --- END PREVIOUS FEEDBACK ---

        Please provide the corrected sequence of function calls:
        """
        

        prompt += feedback_prompt

    try:
        if not os.getenv("GEMINI_API_KEY"):
             raise ValueError("GEMINI_API_KEY not found in environment variables.")

        logging.info("Calling LiteLLM with Gemini 2.0 Flash...")
        response = litellm.completion(
            model="gemini/gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}],
        )
        logging.info("LLM response received.")

        llm_output_content = None
        try:
            llm_output_content = response.choices[0].message.content # type: ignore
            if not isinstance(llm_output_content, str):
                 raise ValueError("LLM response content is not a string or is missing.")
            

            print(llm_output_content)

            logging.debug(f"Raw LLM Output:\n{llm_output_content}")
            cleaned_yaml = re.sub(r'^```yaml\s*|\s*```$', '', llm_output_content, flags=re.MULTILINE).strip()
            generated_operations = yaml.safe_load(cleaned_yaml)

            if not isinstance(generated_operations, list):
                 raise ValueError("LLM did not output a valid YAML list of operations.")

            logging.info(f"LLM generated {len(generated_operations)} operations (parsed from YAML).")
            return generated_operations

        except yaml.YAMLError as e:
            logging.error(f"Failed to parse YAML from LLM response: {e}\nRaw Output:\n{llm_output_content}", exc_info=True)
            raise ValueError(f"LLM output was not valid YAML: {e}")
        except (AttributeError, IndexError, TypeError) as e:
             logging.error(f"Failed to extract content from LLM response structure: {e}", exc_info=True)
             raise ValueError(f"Could not extract content from LLM response: {e}")
        except Exception as e:
             logging.error(f"An unexpected error occurred processing LLM response: {e}", exc_info=True)
             raise

    except Exception as e:
        logging.error(f"Error during LLM operation generation: {e}", exc_info=True)
        raise


# --- Output Validation ---

def validate_schema(df: pl.DataFrame, output_def: FileDefinition) -> str | None:
    """Validates the DataFrame schema against a specific definition's schema."""
    target_schema = output_def.file_schema
    logging.info(f"Validating output DataFrame schema against schema '{target_schema.name}' for '{output_def.path}'...")
    feedback_lines = []
    actual_schema = df.schema
    target_columns = target_schema.columns

    actual_col_names = set(actual_schema.keys())
    target_col_names = set(target_columns.keys())

    missing_cols = target_col_names - actual_col_names
    if missing_cols:
        feedback_lines.append(f"Missing columns required by output schema '{target_schema.name}': {', '.join(sorted(list(missing_cols)))}")

    common_cols = target_col_names.intersection(actual_col_names)
    type_mismatches = []
    for col_name in common_cols:
        actual_type = actual_schema[col_name]
        target_type_str = target_columns[col_name].type
        target_pl_type = POLARS_TYPE_MAP.get(target_type_str)
        actual_type_str = SCHEMA_TYPE_MAP_REVERSE.get(actual_type)

        if not target_pl_type:
            logging.warning(f"Schema validation ({target_schema.name}): Unknown target type '{target_type_str}' for column '{col_name}'. Skipping type check.")
            continue

        if actual_type != target_pl_type:
             logging.warning(f"Schema validation ({target_schema.name}): Type mismatch for column '{col_name}'. Expected Polars type: {target_pl_type} (Schema: '{target_type_str}'), Actual Polars type: {actual_type} (Schema: '{actual_type_str}')")
             type_mismatches.append(f"Column '{col_name}': Expected type '{target_type_str}', Actual type: '{actual_type_str or actual_type}'")

    if type_mismatches:
        feedback_lines.append(f"Type mismatches for schema '{target_schema.name}': {'; '.join(type_mismatches)}")

    if feedback_lines:
        feedback = f"schema validation failed for output '{output_def.path}' (schema '{target_schema.name}'):\n- " + "\n- ".join(feedback_lines)
        logging.warning(feedback)
        return feedback
    else:
        logging.info(f"Output schema validation successful for '{output_def.path}' (schema '{target_schema.name}').")
        return None


# --- Pipeline Definition Loading ---

def load_pipeline_definition(file_path: Path) -> EtlPipeline:
    """Loads and validates the full ETL pipeline definition from YAML."""
    logging.info(f"Loading pipeline definition from: {file_path}")
    if not file_path.exists():
        logging.error(f"Pipeline definition file not found: {file_path}")
        raise FileNotFoundError(f"Pipeline definition file not found: {file_path}")
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        pipeline = EtlPipeline.model_validate(data)
        logging.info("Pipeline definition loaded and validated successfully.")
        return pipeline
    except yaml.YAMLError as e:
        logging.error(f"Error parsing pipeline YAML file: {e}")
        raise
    except Exception as e: # Catch Pydantic validation errors etc.
        logging.error(f"Error validating pipeline definition: {e}")
        raise


# --- Input Data Loading ---

def load_data(input_def: FileDefinition) -> pl.DataFrame:
    """Loads data based on an FileDefinition using Polars."""
    file_path = Path(input_def.path)
    schema = input_def.file_schema
    logging.info(f"Loading data from: {file_path} (Schema: {schema.name})")
    if not file_path.exists():
        logging.error(f"Data file not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}")

    if file_path.suffix.lower() != '.csv':
        raise NotImplementedError("Currently only CSV input format is supported.")

    dtype_map = {}
    schema_cols_dict = schema.columns
    for col_name, col_def in schema_cols_dict.items():
        pl_type = POLARS_TYPE_MAP.get(col_def.type)
        if pl_type:
            dtype_map[col_name] = pl_type
        else:
            logging.warning(f"No Polars type mapping found for schema type: {col_def.type} for column '{col_name}'. Polars will infer.")

    try:
        df = pl.read_csv(file_path, schema_overrides=dtype_map, infer_schema_length=1000 if not dtype_map else 0)
        logging.info(f"Data loaded successfully from {file_path}. Shape: {df.shape}")

        schema_cols = set(schema_cols_dict.keys())
        df_cols = set(df.columns)
        missing_cols = schema_cols - df_cols
        if missing_cols:
            logging.warning(f"Columns defined in schema '{schema.name}' but potentially missing in CSV '{file_path}': {missing_cols}")
        extra_cols = df_cols - schema_cols
        if extra_cols:
             logging.info(f"Columns present in CSV '{file_path}' but not defined in schema '{schema.name}': {extra_cols}")

        return df
    except Exception as e:
        logging.error(f"Error reading CSV file '{file_path}' with Polars: {e}")
        raise


# --- Operation Application ---

def apply_operations(df: pl.DataFrame, operations: Sequence[BaseOperation], pipeline_inputs: Dict[str, FileDefinition]) -> pl.DataFrame:
    """
    Applies the sequence of ETL operations to the Polars DataFrame.
    Requires pipeline_inputs to resolve paths for BindOperation.
    """
    logging.info(f"Applying {len(operations)} operations using Polars...")
    temp_cols = set() # Keep track of temporary columns created by assignation
    current_df = df

    for i, op in enumerate(operations):
        logging.info(f"Applying operation {i+1}/{len(operations)}: {op.operation_type} -> {op.output_column}")
        try:
            # Look up the function based on operation_type
            apply_func = OPERATION_DISPATCHER.get(op.operation_type)

            if apply_func:
                # Special case for bind operation needing pipeline_inputs
                if op.operation_type == 'bind':
                    # We know op is BindOperation here due to dispatcher structure
                    current_df = apply_func(current_df, op, pipeline_inputs) # type: ignore
                else:
                    current_df = apply_func(current_df, op)

                # Track temporary columns created by assignation
                # Ensure op has 'output_column' before checking startswith/endswith
                if hasattr(op, 'output_column') and op.operation_type == 'assignation' and op.output_column.startswith('_') and op.output_column.endswith('_literal'):
                    temp_cols.add(op.output_column)
            else:
                logging.warning(f"Unsupported operation type encountered: {op.operation_type}. Skipping.")

        except Exception as e:
            logging.error(f"Error applying operation {i+1} ({op.operation_type} -> {op.output_column}): {e}", exc_info=True)
            # Add more context to the error message if possible
            error_context = f"Error in operation {i+1} ({op.operation_type} -> {op.output_column}). Details: {e}"
            # You might want to wrap the original exception or just raise a new one with context
            raise ValueError(error_context) from e

    # Remove temporary columns after all operations are done
    if temp_cols:
        cols_to_drop = [col for col in temp_cols if col in current_df.columns]
        if cols_to_drop:
            logging.info(f"Removing temporary columns: {cols_to_drop}")
            current_df = current_df.drop(cols_to_drop)

    logging.info("All operations applied successfully.")
    return current_df


# --- Output Data Saving ---

def save_data(df: pl.DataFrame, output_def: FileDefinition):
    """Saves the transformed Polars DataFrame based on an FileDefinition."""
    output_path = Path(output_def.path)
    output_format = output_def.format # Use format from FileDefinition
    schema = output_def.file_schema
    logging.info(f"Preparing output data for: {output_path} (Schema: {schema.name}, Format: {output_format})")

    output_columns_dict = schema.columns
    output_column_names = list(output_columns_dict.keys())

    final_df = df
    missing_output_cols = []
    present_output_cols = []

    if not output_column_names:
        logging.warning(f"No columns defined in schema '{schema.name}'. Saving all columns to {output_path}.")
    else:
        logging.info(f"Selecting and ordering columns for {output_path}: {output_column_names}")
        for col in output_column_names:
            if col in df.columns:
                present_output_cols.append(col)
            else:
                logging.warning(f"Column '{col}' defined in schema '{schema.name}' but not found in final DataFrame. Skipping for {output_path}.")
                missing_output_cols.append(col)

        if not present_output_cols:
             logging.error(f"No columns specified in the schema '{schema.name}' were found in the transformed data. Output file '{output_path}' will be empty or invalid.")
             final_df = df.select([])
        else:
            final_df = df.select(present_output_cols)

    logging.info(f"Saving data to: {output_path} (Format: {output_format})")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        match output_format:
            case 'csv':
                final_df.write_csv(output_path)
            case 'json':
                with open(output_path, 'w') as f:
                    json.dump(final_df.to_dicts(), f, indent=2)
            case 'parquet':
                final_df.write_parquet(output_path)
            case _:
                # Should be caught by Pydantic, but defensive check
                raise NotImplementedError(f"Format '{output_format}' is not supported.")

        logging.info(f"Data saved successfully to {output_path}.")
        if missing_output_cols:
             logging.warning(f"Note for {output_path}: Columns defined in schema '{schema.name}' but missing from final data: {missing_output_cols}")

    except Exception as e:
        logging.error(f"Error saving output file '{output_path}': {e}")
        raise


# --- Operation Loading ---

def load_operations_from_file(file_path: Path) -> Sequence[BaseOperation]: # Changed return type hint
    """Loads and validates operations structure from a YAML file."""
    logging.info(f"Loading pre-defined operations from: {file_path}")
    if not file_path.exists():
        raise FileNotFoundError(f"Operations file not found: {file_path}")

    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)

        if 'operations' not in data or not isinstance(data['operations'], list):
            raise ValueError(f"YAML file '{file_path}' must contain a top-level 'operations' list.")

        ops_list = data['operations']
        # Basic structural validation (Pydantic validation happens later in main if needed)
        validated_operations = []
        logging.info("Validating loaded operations structure...")
        for i, op_dict in enumerate(ops_list):
             if not isinstance(op_dict, dict) or 'operation_type' not in op_dict:
                 raise ValueError(f"Invalid structure for operation {i+1} in '{file_path}': Missing 'operation_type' or not a dictionary.")
             # Perform Pydantic validation here for robustness
             try:
                 # Use a dummy EtlPipeline context for validation
                 dummy_pipeline_dict = {
                     "inputs": {"dummy": {"path": "dummy", "file_schema": {"name": "Dummy", "columns": {}}}},
                     "outputs": {"dummy": {"path": "dummy", "format": "csv", "file_schema": {"name": "Dummy", "columns": {}}}},
                     "operations": [op_dict]
                 }
                 validated_pipeline = EtlPipeline.model_validate(dummy_pipeline_dict)
                 validated_operations.append(validated_pipeline.operations[0]) # Append the validated Pydantic object
             except Exception as e:
                 logging.error(f"Syntax validation failed for loaded operation {i+1}: {op_dict}. Error: {e}", exc_info=True)
                 raise ValueError(f"Syntax validation failed for loaded operation {i+1}: {e}")

        logging.info(f"Successfully loaded and validated {len(validated_operations)} operations from {file_path}.")
        return validated_operations # Return list of validated Pydantic operation objects

    except yaml.YAMLError as e:
        logging.error(f"Error parsing operations YAML file '{file_path}': {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading or validating operations from '{file_path}': {e}")
        raise


# --- Main Execution ---

MAX_ATTEMPTS = 3

def main():
    parser = argparse.ArgumentParser(description="OpenETLAgent: Apply ETL operations defined in a YAML file using Polars.")
    parser.add_argument("pipeline_file", type=Path, help="Path to the pipeline definition YAML file (containing inputs, outputs, schemas, and optionally operations).")
    # Removed -i and -o arguments
    parser.add_argument("--save-operations", type=Path, default=None, help="Optional path to save the final used operations as a YAML file.")
    parser.add_argument("--use-operations", type=Path, default=None, help="Optional path to a YAML file containing pre-defined operations to use, overriding any in pipeline_file and skipping LLM generation.")

    args = parser.parse_args()

    try:
        # 1. Load Full Pipeline Definition
        pipeline = load_pipeline_definition(args.pipeline_file)

        if not pipeline.inputs:
            logging.error("Pipeline definition must contain at least one input.")
            sys.exit(1)
        if not pipeline.outputs:
            logging.error("Pipeline definition must contain at least one output.")
            sys.exit(1)

        # 2. Load Primary Input Data
        # Convention: Use the first defined input as the primary DataFrame source
        primary_input_key = list(pipeline.inputs.keys())[0]
        primary_input_def = pipeline.inputs[primary_input_key]
        logging.info(f"Using '{primary_input_key}' as the primary input source.")
        df_initial = load_data(primary_input_def)

        # 3. Determine Operations
        final_operations: Optional[Sequence[BaseOperation]] = None # Type hint for clarity
        df_transformed = None # Initialize df_transformed to None
        operations_source = "pipeline YAML" # Default source

        if args.use_operations:
            # Load from external file, overriding pipeline YAML
            operations_source = f"external file ({args.use_operations})"
            final_operations = load_operations_from_file(args.use_operations)
        elif pipeline.operations:
            # Use operations defined within the pipeline YAML
            logging.info("Using operations defined within the pipeline YAML file.")
            final_operations = pipeline.operations # Already validated by load_pipeline_definition
        else:
            # Generate via LLM (Iterative Refinement)
            operations_source = "LLM generation"
            logging.info("No operations found in pipeline YAML or --use-operations. Generating via LLM...")
            feedback = None
            llm_generated_ops = [] # Keep track of the last generated list for saving on failure

            for attempt in range(MAX_ATTEMPTS):
                logging.info(f"--- LLM Generation Attempt {attempt + 1}/{MAX_ATTEMPTS} ---")
                try:
                    # Generate operations list (list of dicts)
                    generated_ops_list = generate_operations_with_llm(pipeline.inputs, pipeline.outputs, previous_feedback=feedback)
                    llm_generated_ops = generated_ops_list # Store the raw list

                    if not generated_ops_list:
                         logging.warning(f"Attempt {attempt + 1}: LLM did not generate any operations.")
                         feedback = "The previous attempt generated no operations. Please generate a valid sequence."
                         continue

                    # Validate syntax by trying to create Pydantic models
                    validated_llm_ops: List[BaseOperation] = [] # Keep as List for appending, but it's compatible with Sequence
                    logging.info("Validating LLM-generated operations syntax...")
                    op_dict_for_validation = None
                    try:
                        # Validate each operation individually within a dummy pipeline context
                        dummy_pipeline_dict = {
                            "inputs": {"dummy": {"path": "dummy", "file_schema": {"name": "Dummy", "columns": {}}}},
                            "outputs": {"dummy": {"path": "dummy", "format": "csv", "file_schema": {"name": "Dummy", "columns": {}}}},
                            "operations": []
                        }
                        for i, op_dict in enumerate(generated_ops_list):
                            op_dict_for_validation = op_dict
                            dummy_pipeline_dict["operations"] = [op_dict] # Validate one at a time
                            validated_pipeline = EtlPipeline.model_validate(dummy_pipeline_dict)
                            validated_llm_ops.append(validated_pipeline.operations[0]) # Append validated Pydantic object
                        logging.info("LLM-generated operations syntax validated successfully.")
                    except Exception as e:
                        logging.error(f"Attempt {attempt + 1}: Syntax validation failed for LLM-generated operation: {op_dict_for_validation}. Error: {e}", exc_info=True)
                        feedback = f"The generated operations failed syntax validation. Error on operation {op_dict_for_validation}: {e}. Please provide a syntactically correct list of operations."
                        continue # Try LLM again

                    # Apply generated & validated operations
                    logging.info(f"Attempt {attempt + 1}: Applying {len(validated_llm_ops)} validated operations...")
                    df_attempt = apply_operations(df_initial.clone(), validated_llm_ops, pipeline.inputs)
                    logging.info(f"Attempt {attempt + 1}: Operations applied.")

                    # Validate output schema(s) for this attempt
                    all_outputs_valid = True
                    output_feedback_list = []
                    for output_key, output_def in pipeline.outputs.items():
                        output_validation_feedback = validate_schema(df_attempt, output_def)
                        if output_validation_feedback is not None:
                            all_outputs_valid = False
                            output_feedback_list.append(output_validation_feedback)

                    if all_outputs_valid:
                        logging.info(f"Attempt {attempt + 1}: All output schemas validated successfully!")
                        final_operations = validated_llm_ops # Store successful Pydantic objects
                        df_transformed = df_attempt # Store successful DataFrame
                        break # Exit LLM loop on success
                    else:
                        feedback = "\n".join(output_feedback_list)
                        logging.warning(f"Attempt {attempt + 1}: Output schema validation failed. Providing feedback to LLM:\n{feedback}")

                except Exception as e:
                     # Catch errors during generation or application within the loop
                     logging.error(f"Attempt {attempt + 1}: Failed during generation or application: {e}", exc_info=True)
                     feedback = f"Attempt {attempt + 1} failed with error: {e}. Please try again, ensuring operations are valid and executable."
                     # Continue to next attempt if possible

            # --- End of LLM Loop ---
            if final_operations is None: # Check if LLM loop succeeded
                 logging.error(f"Failed to generate a valid ETL pipeline via LLM after {MAX_ATTEMPTS} attempts.")
                 if args.save_operations and llm_generated_ops: # Save the raw list from the last attempt
                      logging.info(f"Saving last failed LLM operations attempt to: {args.save_operations}")
                      save_data_ = {'operations': llm_generated_ops, 'final_attempt_feedback': feedback}
                      try:
                          args.save_operations.parent.mkdir(parents=True, exist_ok=True)
                          with open(args.save_operations, 'w') as f:
                              yaml.dump(save_data_, f, sort_keys=False, default_flow_style=False)
                      except Exception as e:
                          logging.error(f"Failed to save last failed LLM operations: {e}")
                 sys.exit(1)

        # 4. Apply Final Operations (if not already done in LLM loop)
        if df_transformed is None:
            if not final_operations:
                 logging.error("No valid operations found from pipeline YAML or external file.")
                 sys.exit(1)
            logging.info(f"Applying {len(final_operations)} operations from {operations_source}...")
            df_transformed = apply_operations(df_initial.clone(), final_operations, pipeline.inputs)
            logging.info("Operations applied.")

        # 5. Validate and Save Outputs
        final_success = True
        for output_key, output_def in pipeline.outputs.items():
            logging.info(f"--- Processing Output: {output_key} ---")
            output_validation_feedback = validate_schema(df_transformed, output_def)
            if output_validation_feedback is None:
                try:
                    save_data(df_transformed, output_def)
                except Exception as e:
                    logging.error(f"Failed to save output '{output_key}' to '{output_def.path}': {e}")
                    final_success = False # Mark failure but continue processing other outputs
            else:
                logging.error(f"Skipping save for output '{output_key}' due to schema validation failure: {output_validation_feedback}")
                final_success = False # Mark failure

        # 6. Save Final Used Operations (if requested and successful)
        if args.save_operations and final_operations and final_success:
            logging.info(f"Saving final used operations ({operations_source}) to: {args.save_operations}")
            try:
                # Convert Pydantic objects back to dicts for saving
                ops_to_save = [op.model_dump() for op in final_operations]
                save_data_ = {'operations': ops_to_save}
                args.save_operations.parent.mkdir(parents=True, exist_ok=True)
                with open(args.save_operations, 'w') as f:
                    yaml.dump(save_data_, f, sort_keys=False, default_flow_style=False)
                logging.info("Final operations saved successfully.")
            except Exception as e:
                logging.error(f"Failed to save final operations: {e}", exc_info=True)
                # Don't fail the whole process if saving ops fails

        if final_success:
            logging.info("ETL process completed successfully for all valid outputs.")
        else:
            logging.error("ETL process completed with errors for one or more outputs.")
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


if __name__ == "__main__":
    main()
