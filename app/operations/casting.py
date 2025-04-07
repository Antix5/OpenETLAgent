import polars as pl
from app.models import CastingOperation, POLARS_TYPE_MAP

def apply_casting(df: pl.DataFrame, op: CastingOperation) -> pl.DataFrame:
    """Applies the casting operation to change a column's data type."""
    if op.input_column not in df.columns:
        raise ValueError(f"CastingOperation: Input column '{op.input_column}' not found in DataFrame columns: {df.columns}")

    target_pl_type = POLARS_TYPE_MAP.get(op.target_type)
    if not target_pl_type:
        raise ValueError(f"CastingOperation: Unsupported target type '{op.target_type}'")

    try:
        return df.with_columns(
            pl.col(op.input_column).cast(target_pl_type).alias(op.output_column)
        )
    except Exception as e:
        # Add more context to the error message
        raise ValueError(f"Failed to cast column '{op.input_column}' (current type: {df[op.input_column].dtype}) to target type '{op.target_type}' (Polars type: {target_pl_type}): {e}")
