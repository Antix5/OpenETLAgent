from app import models
import argparse
import yaml
import polars as pl
from polars import Int64, Utf8, Float64, Boolean, Date # Ensure all needed types are imported
from pathlib import Path
import sys
import logging
import os
import json
from dotenv import load_dotenv
import litellm
from typing import List, Optional, Dict, Sequence, Tuple, Any
import re

litellm.suppress_debug_info = True

# Import models and prompt functions
from app.models import (
    PipelineConfig, PipelineFlow, FileDefinition, BaseOperation, AnyOperation,
    POLARS_TYPE_MAP, SCHEMA_TYPE_MAP_REVERSE,
)
# Import BOTH prompt functions now
from app.instruct_prompt import get_instruct_prompt, get_next_step_prompt
# Import all operation functions
from app.operations.equality import apply_equality
from app.operations.concatenation import apply_concatenation
from app.operations.application import apply_application
from app.operations.asignation import apply_assignation
from app.operations.switching import apply_switching
from app.operations.casting import apply_casting
from app.operations.arithmetic import apply_arithmetic
from app.operations.comparison import apply_comparison
from app.operations.bind import apply_bind
from app.operations.fold import apply_fold
from app.operations.unfold import apply_unfold

# Load environment variables from .env file
load_dotenv()

# Check environment variable to control LLM output printing for debugging
SHOW_MODEL_OUTPUT = os.getenv('SHOW_MODEL_OUTPUT', 'False').lower() == 'true'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Changed level to WARNING

# Define a filter to suppress LiteLLM INFO messages
class SuppressLiteLLMInfoFilter(logging.Filter):
    def filter(self, record):
        # Suppress INFO messages specifically from loggers starting with 'litellm'
        return not (record.name.startswith('litellm') and record.levelno == logging.INFO)

# Add the filter to the root logger's handlers
# This assumes basicConfig added a handler to the root logger
# Add filter only if handlers exist to avoid errors if basicConfig fails or is changed
if logging.root.handlers:
    for handler in logging.root.handlers:
        handler.addFilter(SuppressLiteLLMInfoFilter())
else:
    logging.warning("No handlers found on root logger after basicConfig. LiteLLM INFO filter not applied.")


# Define pipelines directory
PIPELINES_DIR = Path("pipelines")

# --- Operation Dispatcher (Unchanged) ---
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
    'fold': apply_fold,
    'unfold': apply_unfold,
}

# --- LLM Generation (Step-by-Step) ---
MAX_STEPS = 35 # Maximum number of operations to generate per pipeline
MAX_LLM_ATTEMPTS_PER_STEP = 6 # Retries for LLM call or validation failure

def generate_pipeline_step_by_step(
    config: PipelineConfig,
    target_output_key: str,
    target_output_def: FileDefinition
) -> PipelineFlow:
    """
    Generates a pipeline flow step-by-step using the LLM.
    """
    logging.info(f"Starting step-by-step generation for output '{target_output_key}'...")

    # 1. Determine Initial Source (Ask LLM once at the beginning)
    logging.info("Determining initial source via LLM...")
    initial_prompt = get_instruct_prompt(target_output_def=target_output_def, inputs=config.inputs)
    initial_prompt += "\n\nBased on the inputs and target output, what is the single best `source` input key to start the pipeline with? Respond with ONLY the input key string (e.g., 'orders_input')."

    source_key = None
    for attempt in range(MAX_LLM_ATTEMPTS_PER_STEP):
        try:
            if not os.getenv("GEMINI_API_KEY"):
                   raise ValueError("GEMINI_API_KEY not found.")
            response = litellm.completion(
                    model=os.getenv("MODEL", "gemini-2.0-flash"),
                    messages=[{"role": "user", "content": initial_prompt}],
                    max_tokens=500
                 )
            # Safely access content
            source_key_content = None
            if response and response.choices and response.choices[0].message: # type: ignore
                    source_key_content = response.choices[0].message.content # type: ignore[attr-defined] # type: ignore[union-attr]

            if isinstance(source_key_content, str):
                source_key_raw = source_key_content.strip().replace("'", "").replace('"', '')
                if source_key_raw in config.inputs:
                    source_key = source_key_raw
                    logging.info(f"LLM chose initial source: '{source_key}'")
                    break
                else:
                    logging.warning(f"LLM returned invalid source key '{source_key_raw}' (Attempt {attempt+1}). Retrying...")
                    initial_prompt += f"\n\nInvalid source '{source_key_raw}' provided. Choose from {list(config.inputs.keys())}."
            else:
                logging.warning(f"LLM response content was not a string (Attempt {attempt+1}). Retrying...")
                initial_prompt += f"\n\nInvalid response format received. Please respond with only the source key string."
        except Exception as e:
              logging.error(f"Error getting initial source from LLM (Attempt {attempt+1}): {e}")

    if not source_key:
         source_key = list(config.inputs.keys())[0]
         logging.warning(f"LLM failed to provide a valid source key. Falling back to first input: '{source_key}'")

    # 2. Load Initial Data and Schema
    source_def = config.inputs[source_key]
    current_df = load_data(source_def)
    current_state_header : str = current_df.head(3).write_json()
    operations_history: List[Dict[str, Any]] = []
    validated_operations: List[AnyOperation] = []

    # 3. Iterative Step Generation Loop
    for step in range(MAX_STEPS):
        logging.info(f"--- Generating Step {step + 1}/{MAX_STEPS} for '{target_output_key}' ---")
        current_schema_dict = {k: str(v) for k, v in current_df.schema.items()}
        feedback = None

        for attempt in range(MAX_LLM_ATTEMPTS_PER_STEP):
            logging.info(f"--- LLM Step Attempt {attempt + 1}/{MAX_LLM_ATTEMPTS_PER_STEP} ---")
            try:
                # a. Get next operation suggestion
                next_op_prompt = get_next_step_prompt(
                    current_schema_dict=current_schema_dict,
                    target_output_def=target_output_def,
                    inputs=config.inputs,
                    operations_history=operations_history,
                    current_state_header=current_state_header,
                    previous_feedback=feedback
                )

                if not os.getenv("GEMINI_API_KEY"):
                    raise ValueError("GEMINI_API_KEY not found.")
                response = litellm.completion(
                    model=os.getenv("MODEL", "gemini-2.0-flash"),
                    messages=[{"role": "user", "content": next_op_prompt}]
                )

                # Safely access content
                llm_output_content = None
                if response and response.choices and response.choices[0].message: # type: ignore
                     llm_output_content = response.choices[0].message.content # type: ignore

                if not isinstance(llm_output_content, str):
                     raise ValueError("LLM response content was not a string.")
                
                match = re.search(r"^```yaml\s*(.*?)\s*```$", llm_output_content, flags=re.M | re.S)
                cleaned_yaml = match.group(1).strip() if match else ""

                # Conditional printing for debugging LLM output
                if SHOW_MODEL_OUTPUT:
                    print(f"\n--- RAW LLM OUTPUT (Step {step+1}, Attempt {attempt+1}) ---")
                    print(llm_output_content)
                    print("----------------------------------------------------\n")
                    # Also print the cleaned version right before parsing
                    # Use a separate variable for printing to avoid altering the one used for parsing

                    print(f"--- CLEANED YAML (Ready for Parsing) ---")
                    print(cleaned_yaml)
                    print("----------------------------------------\n")

                # Proceed with cleaning and parsing
                
                next_op_dict_raw = yaml.safe_load(cleaned_yaml)

                try:
                    next_op_dict = next_op_dict_raw[0]
                except Exception:
                    raise ValueError("Could not parse the yaml you provided, make sure you didn't miss '- ' in front of operation type")


                if SHOW_MODEL_OUTPUT:
                    print("parsed yaml : ", next_op_dict_raw)

                if not isinstance(next_op_dict, dict) or 'operation_type' not in next_op_dict:
                     raise ValueError("LLM did not output a valid YAML dictionary with 'operation_type'.")
                
                # b. Check for 'done' signal
                if next_op_dict['operation_type'] == 'done':
                    logging.info("LLM signaled completion ('operation_type: done').")
                    final_validation_feedback = validate_schema(current_df, target_output_def, target_output_key)
                    if final_validation_feedback:
                         logging.warning(f"LLM signaled 'done', but final schema validation failed: {final_validation_feedback}")
                         feedback = f"You signaled 'done', but the current schema does not match the target. Validation failed: {final_validation_feedback}. Please provide the next operation."
                         continue
                    else:
                         logging.info("Final schema matches target. Pipeline generation complete.")
                         return PipelineFlow(source=source_key, operations=validated_operations)

                # c. Validate Syntax
                validated_op: Optional[AnyOperation] = None
                try:
                    # Attempt validation against each possible operation type
                    for operation_type in [
                        models.EqualityOperation,
                        models.ConcatenationOperation,
                        models.ApplicationOperation,
                        models.SwitchingOperation,
                        models.AssignationOperation,
                        models.CastingOperation,
                        models.ArithmeticOperation,
                        models.ComparisonOperation,
                        models.BindOperation,
                        models.FoldOperation,
                        models.UnfoldOperation,
                    ]:
                        try:
                            # Use model_validate instead of deprecated parse_obj
                            validated_op = operation_type.model_validate(next_op_dict)
                            op_type_str = getattr(validated_op, 'operation_type', 'UnknownType')
                            logging.info(f"Step {step+1}: Syntax validated for operation '{op_type_str}'.")
                            break  # Exit loop if validation succeeds
                        except Exception:
                            continue  # Try the next operation type
                    else:
                        # If none of the operation types validated, raise an error
                        raise ValueError("No matching operation type found in AnyOperation union.")
                except Exception as e:
                    logging.error(f"Step {step+1}, Attempt {attempt+1}: Syntax validation failed: {next_op_dict}. Error: {e}", exc_info=True) # Verbose for debug
                    continue

                # d. Apply Operation (Trial Run) - Ensure validated_op is not None
                if validated_op:
                    op_type_str = getattr(validated_op, 'operation_type', 'UnknownType') # Get type again safely
                    logging.info(f"Step {step+1}, Attempt {attempt+1}: Applying operation '{op_type_str}'...")
                    # Pass validated_op which is known to be AnyOperation here
                    df_next = apply_operations(current_df.clone(), [validated_op], config.inputs) # type: ignore[list-item] # Pylance struggles with Sequence[Base] vs List[Union]
                    logging.info(f"Step {step+1}, Attempt {attempt+1}: Operation applied successfully.")

                    if SHOW_MODEL_OUTPUT:
                        print("the current table look like this :")
                        print(current_df)

                    # Update state and break inner loop
                    current_df = df_next
                    current_state_header : str = current_df.head(3).write_json()
                    operations_history.append(next_op_dict)
                    validated_operations.append(validated_op) # Append validated op
                    feedback = None
                    break # Go to the next step
                else:
                     # Should not happen if validation passed, but defensive check
                     logging.error(f"Step {step+1}, Attempt {attempt+1}: validated_op is None after validation passed. Skipping.")
                     feedback = "Internal error: Operation validated but resulted in None."
                     continue


            except Exception as e:
                 logging.error(f"Step {step+1}, Attempt {attempt+1}: Failed: {e}", exc_info=True)
                 feedback = f"Attempt {attempt + 1} for step {step+1} failed with error: {e}. Please try again."

        if feedback is not None:
             raise RuntimeError(f"Failed to generate a valid operation for step {step + 1} after {MAX_LLM_ATTEMPTS_PER_STEP} attempts for output '{target_output_key}'. Last error feedback: {feedback}")

    raise RuntimeError(f"Pipeline generation exceeded maximum steps ({MAX_STEPS}) for output '{target_output_key}'.")


# --- Output Validation (Unchanged) ---
def validate_schema(df: pl.DataFrame, output_def: FileDefinition, output_key: str) -> str | None:
    """Validates the DataFrame schema against a specific output definition's schema."""
    target_schema = output_def.file_schema
    logging.info(f"Validating schema for output '{output_key}' ('{target_schema.name}') against path '{output_def.path}'...")
    feedback_lines = []
    actual_schema = df.schema
    target_columns = target_schema.columns

    actual_col_names = set(actual_schema.keys())
    target_col_names = set(target_columns.keys())

    if target_col_names:
        missing_cols = target_col_names - actual_col_names
        if missing_cols:
            feedback_lines.append(f"Missing columns required by output schema '{target_schema.name}': {', '.join(sorted(list(missing_cols)))}")
        common_cols = target_col_names.intersection(actual_col_names)
    else:
        logging.warning(f"Schema validation ({target_schema.name}): No columns defined in target schema. Will not check for missing columns.")
        common_cols = actual_col_names

    type_mismatches = []
    for col_name in common_cols:
        if col_name in target_columns:
            actual_type = actual_schema[col_name]
            target_type_str = target_columns[col_name].type
            target_pl_type = POLARS_TYPE_MAP.get(target_type_str)
            actual_type_str = SCHEMA_TYPE_MAP_REVERSE.get(actual_type)

            if not target_pl_type:
                logging.warning(f"Schema validation ({target_schema.name}): Unknown target type '{target_type_str}' for column '{col_name}'. Skipping type check.")
                continue
            if target_type_str == 'positive integer' and actual_type == Int64:
                 logging.debug(f"Schema validation ({target_schema.name}): Allowing Int64 for target 'positive integer' column '{col_name}'.")
                 continue
            if actual_type != target_pl_type:
                logging.warning(f"Schema validation ({target_schema.name}): Type mismatch for column '{col_name}'. Expected Polars type: {target_pl_type} (Schema: '{target_type_str}'), Actual Polars type: {actual_type} (Schema: '{actual_type_str}')")
                type_mismatches.append(f"Column '{col_name}': Expected type '{target_type_str}', Actual type: '{actual_type_str or actual_type}'")

    if type_mismatches:
        feedback_lines.append(f"Type mismatches for schema '{target_schema.name}': {'; '.join(type_mismatches)}")

    if feedback_lines:
        feedback = f"Schema validation failed for output '{output_key}' (schema '{target_schema.name}'):\n- " + "\n- ".join(feedback_lines)
        logging.warning(feedback)
        return feedback
    else:
        logging.info(f"Output schema validation successful for '{output_key}' (schema '{target_schema.name}').")
        return None

# --- Config Loading (Unchanged) ---
def load_pipeline_config(file_path: Path) -> PipelineConfig:
    """Loads and validates the central pipeline configuration from YAML."""
    logging.info(f"Loading pipeline configuration from: {file_path}")
    if not file_path.exists():
        logging.error(f"Pipeline configuration file not found: {file_path}")
        raise FileNotFoundError(f"Pipeline configuration file not found: {file_path}")
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        config = PipelineConfig.model_validate(data)
        logging.info("Pipeline configuration loaded and validated successfully.")
        if not config.inputs:
             raise ValueError("Configuration must define at least one input.")
        if not config.outputs:
             raise ValueError("Configuration must define at least one output.")
        return config
    except yaml.YAMLError as e:
        logging.error(f"Error parsing pipeline config YAML file: {e}")
        raise
    except Exception as e:
        logging.error(f"Error validating pipeline configuration: {e}")
        raise

# --- Input Data Loading (Unchanged) ---
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
        type_key = col_def.type
        pl_type = POLARS_TYPE_MAP.get(type_key)
        if pl_type:
            dtype_map[col_name] = pl_type
        else:
            logging.warning(f"No Polars type mapping found for schema type: {type_key} for column '{col_name}'. Polars will infer.")
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

# --- Operation Application (Unchanged) ---
def apply_operations(df: pl.DataFrame, operations: Sequence[BaseOperation], pipeline_inputs: Dict[str, FileDefinition]) -> pl.DataFrame:
    """Applies the sequence of ETL operations to the Polars DataFrame."""
    logging.info(f"Applying {len(operations)} operations using Polars...")
    temp_cols = set()
    current_df = df
    for i, op in enumerate(operations):
        op_type_str = getattr(op, 'operation_type', 'UnknownType')
        output_col_str = getattr(op, 'output_column', 'N/A')
        logging.info(f"Applying operation {i+1}/{len(operations)}: {op_type_str} -> {output_col_str}")
        try:
            apply_func = OPERATION_DISPATCHER.get(op_type_str)
            if apply_func:
                if op_type_str == 'bind':
                    current_df = apply_func(current_df, op, pipeline_inputs) # type: ignore
                else:
                    current_df = apply_func(current_df, op)
                if output_col_str and isinstance(output_col_str, str) and \
                   op_type_str == 'assignation' and \
                   output_col_str.startswith('_') and output_col_str.endswith('_'):
                    temp_cols.add(output_col_str)
            else:
                logging.warning(f"Unsupported operation type encountered: {op_type_str}. Skipping.")
        except Exception as e:
            logging.error(f"Error applying operation {i+1} ({op_type_str} -> {output_col_str}): {e}", exc_info=True)
            error_context = f"Error in operation {i+1} ({op_type_str} -> {output_col_str}). Details: {e}"
            raise ValueError(error_context) from e
    if temp_cols:
        cols_to_drop = [col for col in temp_cols if col in current_df.columns]
        if cols_to_drop:
            logging.info(f"Removing temporary columns: {cols_to_drop}")
            current_df = current_df.drop(cols_to_drop)
    logging.info("All operations applied successfully.")
    return current_df

# --- Output Data Saving (Unchanged) ---
def save_data(df: pl.DataFrame, output_def: FileDefinition, output_key: str):
    """Saves the transformed Polars DataFrame based on an output definition."""
    output_path = Path(output_def.path)
    output_format = output_def.format
    schema = output_def.file_schema
    logging.info(f"Preparing output data for '{output_key}' to: {output_path} (Schema: {schema.name}, Format: {output_format})")
    output_columns_dict = schema.columns
    output_column_names = list(output_columns_dict.keys())
    final_df = df
    missing_output_cols = []
    present_output_cols = []
    if output_column_names:
        logging.info(f"Selecting and ordering columns for '{output_key}' based on schema '{schema.name}': {output_column_names}")
        for col in output_column_names:
            if col in df.columns:
                present_output_cols.append(col)
            else:
                logging.warning(f"Column '{col}' defined in schema '{schema.name}' for output '{output_key}' but not found in final DataFrame. Skipping.")
                missing_output_cols.append(col)
        if not present_output_cols:
             logging.error(f"No columns specified in the schema '{schema.name}' were found in the transformed data for output '{output_key}'. Output file '{output_path}' will be empty or invalid.")
             final_df = df.select([])
        else:
            final_df = df.select(present_output_cols)
    else:
        logging.warning(f"No columns defined in schema '{schema.name}' for output '{output_key}'. Saving all columns present in DataFrame to {output_path}.")
    logging.info(f"Saving data for '{output_key}' to: {output_path} (Format: {output_format})")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        match output_format:
            case 'csv':
                final_df.write_csv(output_path)
            case 'json':
                for col_name in final_df.columns:
                    if final_df[col_name].dtype == pl.Date:
                        logging.debug(f"Converting Date column '{col_name}' to string for JSON output.")
                        final_df = final_df.with_columns(pl.col(col_name).dt.strftime('%Y-%m-%d'))
                with open(output_path, 'w') as f:
                    json.dump(final_df.to_dicts(), f, indent=2)
            case 'parquet':
                final_df.write_parquet(output_path)
            case _:
                raise NotImplementedError(f"Format '{output_format}' is not supported.")
        logging.info(f"Data saved successfully for '{output_key}' to {output_path}.")
        if missing_output_cols:
             logging.warning(f"Note for {output_path}: Columns defined in schema '{schema.name}' but missing from final data: {missing_output_cols}")
    except Exception as e:
        logging.error(f"Error saving output file '{output_path}' for output '{output_key}': {e}")
        raise

# --- Pipeline Flow Loading (Unchanged) ---
def load_pipeline_flow(file_path: Path) -> PipelineFlow:
    """Loads and validates a pipeline flow (source + operations) from a YAML file."""
    logging.info(f"Loading pipeline flow from: {file_path}")
    if not file_path.exists():
        raise FileNotFoundError(f"Pipeline flow file not found: {file_path}")
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        flow = PipelineFlow.model_validate(data)
        logging.info(f"Successfully loaded and validated pipeline flow from {file_path} (Source: {flow.source}, Operations: {len(flow.operations)}).")
        return flow
    except yaml.YAMLError as e:
        logging.error(f"Error parsing pipeline flow YAML file '{file_path}': {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading or validating pipeline flow from '{file_path}': {e}")
        raise

# --- Save Pipeline Flow (Unchanged) ---
def save_pipeline_flow(flow: PipelineFlow, file_path: Path):
    """Saves a PipelineFlow object to a YAML file."""
    logging.info(f"Saving generated pipeline flow to: {file_path}")
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        flow_dict = flow.model_dump(mode='python')
        with open(file_path, 'w') as f:
            yaml.dump(flow_dict, f, sort_keys=False, default_flow_style=False)
        logging.info(f"Pipeline flow saved successfully to {file_path}.")
    except Exception as e:
        logging.error(f"Failed to save pipeline flow to {file_path}: {e}", exc_info=True)
        raise

# --- Main Execution (Revised with Step-by-Step Generation) ---
def main():
    parser = argparse.ArgumentParser(description="OpenETLAgent: Run ETL pipelines defined by a central config and individual flow files.")
    parser.add_argument("config_file", type=Path, help="Path to the central pipeline configuration YAML file (defining inputs and outputs).")
    args = parser.parse_args()

    overall_success = True

    try:
        # 1. Load Central Configuration
        config = load_pipeline_config(args.config_file)

        # 2. Ensure pipelines directory exists
        PIPELINES_DIR.mkdir(parents=True, exist_ok=True)

        # 3. Iterate through each defined output
        for output_key, output_def in config.outputs.items():
            logging.info(f"--- Processing Pipeline for Output: '{output_key}' ---")
            pipeline_success = False
            flow_file_path = PIPELINES_DIR / f"{output_key}.yaml"
            final_flow: Optional[PipelineFlow] = None
            final_df: Optional[pl.DataFrame] = None
            operations_source = "unknown"

            try:
                # 4. Load or Generate Pipeline Flow
                if flow_file_path.exists():
                    logging.info(f"Found existing pipeline flow file: {flow_file_path}")
                    try:
                        final_flow = load_pipeline_flow(flow_file_path)
                        operations_source = f"file ({flow_file_path.name})"
                    except Exception as e:
                         logging.error(f"Failed to load/validate existing flow file {flow_file_path}: {e}. Skipping output '{output_key}'.", exc_info=True)
                         overall_success = False
                         continue
                else:
                    # --- Step-by-Step Generation ---
                    logging.info(f"Pipeline flow file not found: {flow_file_path}. Starting step-by-step generation...")
                    operations_source = "LLM step-by-step generation"
                    try:
                        final_flow = generate_pipeline_step_by_step(config, output_key, output_def)
                        # Save the successfully generated flow
                        save_pipeline_flow(final_flow, flow_file_path)
                        logging.info(f"Successfully generated and saved pipeline flow for '{output_key}'.")
                    except Exception as e:
                         logging.error(f"Step-by-step generation failed for output '{output_key}': {e}", exc_info=True)
                         overall_success = False
                         continue # Skip to next output

                # 5. Execute the Loaded/Generated Pipeline Flow
                if final_flow:
                    source_key = final_flow.source
                    operations = final_flow.operations
                    if source_key not in config.inputs:
                        raise ValueError(f"Source key '{source_key}' defined in pipeline flow '{flow_file_path.name}' not found in config inputs.")

                    source_def = config.inputs[source_key]
                    df_initial = load_data(source_def) # Load data for execution

                    logging.info(f"Applying {len(operations)} operations from {operations_source} for output '{output_key}'...")
                    # Add type ignore for apply_operations call
                    df_transformed = apply_operations(df_initial.clone(), operations, config.inputs) # type: ignore[arg-type]
                    logging.info(f"Operations applied for '{output_key}'.")
                    final_df = df_transformed # Store the result for saving

                # 6. Validate and Save Final Data (if execution happened)
                if final_df is not None:
                    validation_feedback = validate_schema(final_df, output_def, output_key)
                    if validation_feedback is None:
                        save_data(final_df, output_def, output_key)
                        pipeline_success = True
                    else:
                        logging.error(f"Schema validation failed for pipeline '{output_key}' after execution:\n{validation_feedback}")
                        overall_success = False
                elif final_flow: # If flow existed but execution failed somehow before final_df was set
                     overall_success = False

            except FileNotFoundError as e:
                logging.error(f"Error processing pipeline for '{output_key}': Required file not found: {e}")
                overall_success = False
            except NotImplementedError as e:
                 logging.error(f"Error processing pipeline for '{output_key}': Feature not implemented: {e}")
                 overall_success = False
            except Exception as e:
                logging.error(f"Failed to process pipeline for output '{output_key}': {e}", exc_info=True)
                overall_success = False

            if not pipeline_success and operations_source != "LLM step-by-step generation": # Only mark overall failure if a pre-defined pipeline failed validation/saving
                 overall_success = False

            logging.info(f"--- Finished Processing Output: '{output_key}' ---")

        # --- End of Output Loop ---

        if overall_success:
            logging.info("All requested pipeline flows processed successfully.")
        else:
            logging.error("One or more pipeline flows failed during processing.")
            sys.exit(1)

    except FileNotFoundError as e:
        logging.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    litellm.suppress_debug_info = True
    main()
