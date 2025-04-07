import polars as pl
from pathlib import Path
from typing import Dict
import logging
from app.models import BindOperation, FileDefinition, POLARS_TYPE_MAP

def apply_bind(df: pl.DataFrame, op: BindOperation, pipeline_inputs: Dict[str, FileDefinition]) -> pl.DataFrame:
    """Applies the bind (join) operation."""
    # Resolve right_file_path against the paths defined in pipeline_inputs
    resolved_right_path_str = None
    if op.right_file_path in pipeline_inputs:
        resolved_right_path_str = pipeline_inputs[op.right_file_path].path
        logging.info(f"Binding with input key '{op.right_file_path}' resolved to path '{resolved_right_path_str}' on {op.left_on} = {op.right_on}")
    else:
        # Assume it's a direct path relative to project root
        resolved_right_path_str = op.right_file_path
        logging.info(f"Binding with direct path '{resolved_right_path_str}' on {op.left_on} = {op.right_on}")

    right_file_path = Path(resolved_right_path_str)
    if not right_file_path.exists():
        raise FileNotFoundError(f"BindOperation: Right file not found at resolved path '{resolved_right_path_str}' (from operation value '{op.right_file_path}')")

    # Prepare schema for reading the right DataFrame
    right_dtype_map = {}
    for col_name, type_str in op.right_schema_columns.items():
        pl_type = POLARS_TYPE_MAP.get(type_str)
        if pl_type:
            right_dtype_map[col_name] = pl_type
        else:
            logging.warning(f"BindOperation: No Polars type mapping for right schema type '{type_str}' for column '{col_name}'. Polars will infer.")

    # Read the right DataFrame
    try:
        df_right = pl.read_csv(right_file_path, schema_overrides=right_dtype_map, infer_schema_length=1000 if not right_dtype_map else 0)
        logging.debug(f"Right DataFrame loaded. Shape: {df_right.shape}, Columns: {df_right.columns}")
    except Exception as e:
        raise ValueError(f"BindOperation: Error reading right file '{right_file_path}': {e}") from e

    # Validate join keys and columns to add
    if op.left_on not in df.columns:
        raise ValueError(f"BindOperation: Left join key '{op.left_on}' not found in current DataFrame columns: {df.columns}")
    if op.right_on not in df_right.columns:
        raise ValueError(f"BindOperation: Right join key '{op.right_on}' not found in right DataFrame ('{right_file_path}') columns: {df_right.columns}")
    missing_add_cols = [col for col in op.columns_to_add if col not in df_right.columns]
    if missing_add_cols:
        raise ValueError(f"BindOperation: Columns specified in 'columns_to_add' not found in right DataFrame: {missing_add_cols}")

    # Perform the join
    try:
        # Select only necessary columns from the right DataFrame (join key + columns to add)
        cols_to_select_from_right = list(set([op.right_on] + op.columns_to_add))
        df_right_selected = df_right.select(cols_to_select_from_right)

        # Perform the join
        joined_df = df.join(df_right_selected, left_on=op.left_on, right_on=op.right_on, how=op.how)
        logging.debug(f"DataFrame shape after join: {joined_df.shape}")
        return joined_df
    except Exception as e:
        raise ValueError(f"BindOperation: Error during join operation: {e}") from e
