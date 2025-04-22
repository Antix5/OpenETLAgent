import polars as pl
import logging
from app.models import FoldOperation
from typing import List

def apply_fold(df: pl.DataFrame, op: FoldOperation) -> pl.DataFrame:
    """
    Applies the fold (melt) operation to the DataFrame.
    Transforms the DataFrame from wide to long format.
    """
    # Validate id_columns exist
    missing_id_cols = [col for col in op.id_columns if col not in df.columns]
    if missing_id_cols:
        raise ValueError(f"FoldOperation: ID columns not found in DataFrame: {missing_id_cols}. Available: {df.columns}")

    # Validate value_columns exist
    missing_value_cols = [col for col in op.value_columns if col not in df.columns]
    if missing_value_cols:
        raise ValueError(f"FoldOperation: Value columns not found in DataFrame: {missing_value_cols}. Available: {df.columns}")

    logging.debug(f"Applying fold: id_columns={op.id_columns}, value_columns={op.value_columns}, "
                  f"key_column_name='{op.key_column_name}', value_column_name='{op.value_column_name}'")

    try:
        return df.melt(
            id_vars=op.id_columns,
            value_vars=op.value_columns,
            variable_name=op.key_column_name,
            value_name=op.value_column_name
        )
    except Exception as e:
        raise ValueError(f"Failed to apply fold operation: {e}") from e
