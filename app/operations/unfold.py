import polars as pl
import logging
from app.models import UnfoldOperation
from typing import List

def apply_unfold(df: pl.DataFrame, op: UnfoldOperation) -> pl.DataFrame:
    """
    Applies the unfold (pivot) operation to the DataFrame.
    Transforms the DataFrame from long to wide format.
    """
    # Validate index_columns exist
    missing_index_cols = [col for col in op.index_columns if col not in df.columns]
    if missing_index_cols:
        raise ValueError(f"UnfoldOperation: Index columns not found in DataFrame: {missing_index_cols}. Available: {df.columns}")

    # Validate key_column exists
    if op.key_column not in df.columns:
        raise ValueError(f"UnfoldOperation: Key column '{op.key_column}' not found in DataFrame. Available: {df.columns}")

    # Validate value_column exists
    if op.value_column not in df.columns:
        raise ValueError(f"UnfoldOperation: Value column '{op.value_column}' not found in DataFrame. Available: {df.columns}")

    logging.debug(f"Applying unfold: index_columns={op.index_columns}, key_column='{op.key_column}', value_column='{op.value_column}'")

    try:
        # Note: Polars pivot requires aggregation if there are duplicate keys per index.
        # Defaulting to 'first' aggregation. User might need to preprocess if other aggregation is needed.
        # Consider adding an 'aggregate_function' parameter to the model if more control is needed.
        return df.pivot(
            index=op.index_columns, # Correct parameter for index columns
            on=op.key_column,       # Correct parameter for the column to pivot on (becomes new headers)
            values=op.value_column,
            aggregate_function="first" # Default aggregation
        )
    except Exception as e:
        raise ValueError(f"Failed to apply unfold (pivot) operation: {e}. Ensure index/key columns uniquely identify rows or consider aggregation needs.") from e
