import polars as pl
from app.models import EqualityOperation

def apply_equality(df: pl.DataFrame, op: EqualityOperation) -> pl.DataFrame:
    """Applies the equality operation (renames or duplicates a column)."""
    if op.input_column not in df.columns:
        raise ValueError(f"EqualityOperation: Input column '{op.input_column}' not found in DataFrame columns: {df.columns}")
    return df.with_columns(
        pl.col(op.input_column).alias(op.output_column)
    )
