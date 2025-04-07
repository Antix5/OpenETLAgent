import polars as pl
from polars import Boolean
import logging
from app.models import SwitchingOperation

def apply_switching(df: pl.DataFrame, op: SwitchingOperation) -> pl.DataFrame:
    """Applies the switching operation based on a condition column."""
    if op.condition_column not in df.columns:
        raise ValueError(f"SwitchingOperation: Condition column '{op.condition_column}' not found in DataFrame columns: {df.columns}")

    # Determine if true/false values are column names or literals
    true_val_expr = pl.col(op.true_column) if op.true_column in df.columns else pl.lit(op.true_column)
    false_val_expr = pl.col(op.false_column) if op.false_column in df.columns else pl.lit(op.false_column)

    # Handle condition column type
    condition_col_expr = pl.col(op.condition_column)
    if df[op.condition_column].dtype != Boolean:
        logging.warning(f"SwitchingOperation: Condition column '{op.condition_column}' is not boolean type ({df[op.condition_column].dtype}). Attempting cast.")
        try:
            condition_col_expr = pl.col(op.condition_column).cast(Boolean)
        except Exception as e:
            raise ValueError(f"Failed to cast condition column '{op.condition_column}' (type: {df[op.condition_column].dtype}) to boolean: {e}")

    # Build the when/then/otherwise expression
    switch_expr = pl.when(condition_col_expr).then(true_val_expr).otherwise(false_val_expr).alias(op.output_column)
    return df.with_columns(switch_expr)
