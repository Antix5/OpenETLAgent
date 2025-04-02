import argparse
import yaml
import polars as pl
from polars import Utf8, Int64, Float64, Boolean # Explicit Polars types
from pathlib import Path
import sys
import logging
import warnings # To suppress Polars experimental warning for map_elements if needed

# Assuming models.py is in the same directory or PYTHONPATH is set correctly
try:
    from models import (
        EtlPipeline, EqualityOperation, ConcatenationOperation,
        ApplicationOperation, SwitchingOperation, AssignationOperation,
        CastingOperation, FileSchema # Added FileSchema import
    )
except ImportError:
    # Simple fallback for running directly if app module isn't installed
    sys.path.insert(0, str(Path(__file__).parent.resolve()))
    from models import (
        EtlPipeline, EqualityOperation, ConcatenationOperation,
        ApplicationOperation, SwitchingOperation, AssignationOperation,
        CastingOperation, FileSchema # Added FileSchema import
    )


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


def load_pipeline_definition(file_path: Path) -> EtlPipeline:
    """Loads and validates the pipeline definition YAML."""
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
                    map_expr = pl.struct(op.input_columns).map_elements(
                        lambda row_struct: lambda_func(row_struct), # Pass the struct directly
                        return_dtype=Float64 # TODO: Infer return type or make it configurable? Defaulting to Float64 for now.
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
                if op.true_column not in current_df.columns:
                    raise ValueError(f"SwitchingOperation: True column '{op.true_column}' not found.")
                if op.false_column not in current_df.columns:
                    raise ValueError(f"SwitchingOperation: False column '{op.false_column}' not found.")

                # Ensure condition column is boolean (Polars is stricter)
                if current_df[op.condition_column].dtype != Boolean:
                     logging.warning(f"SwitchingOperation: Condition column '{op.condition_column}' is not boolean type ({current_df[op.condition_column].dtype}). Attempting cast.")
                     try:
                         current_df = current_df.with_columns(pl.col(op.condition_column).cast(Boolean))
                     except Exception as e:
                         raise ValueError(f"Failed to cast condition column '{op.condition_column}' to boolean: {e}")

                switch_expr = pl.when(pl.col(op.condition_column)) \
                                .then(pl.col(op.true_column)) \
                                .otherwise(pl.col(op.false_column)) \
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


def main():
    parser = argparse.ArgumentParser(description="OpenETLAgent: Apply ETL operations based on a YAML definition using Polars.")
    parser.add_argument("pipeline_file", type=Path, help="Path to the pipeline definition YAML file.")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Path to the input data file (e.g., CSV).")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Path for the output data file.")

    args = parser.parse_args()

    try:
        # 1. Load Pipeline Definition
        pipeline = load_pipeline_definition(args.pipeline_file)

        # 2. Load Input Data
        df = load_input_data(args.input, pipeline.input_schema) # Pass FileSchema object

        # 3. Apply Operations
        df_transformed = apply_operations(df, pipeline.operations)

        # 4. Save Output Data
        save_output_data(df_transformed, args.output, pipeline.output_schema) # Pass FileSchema object

        logging.info("ETL process completed successfully using Polars.")

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
