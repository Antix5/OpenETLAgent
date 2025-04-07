import polars as pl
from polars import Utf8
from app.models import ConcatenationOperation

def apply_concatenation(df: pl.DataFrame, op: ConcatenationOperation) -> pl.DataFrame:
    """Applies the concatenation operation."""
    missing_inputs = [col for col in op.input_columns if col not in df.columns]
    if missing_inputs:
        raise ValueError(f"ConcatenationOperation: Input columns not found: {missing_inputs} in DataFrame columns: {df.columns}")
    concat_expr = pl.concat_str(
        [pl.col(c).cast(Utf8) for c in op.input_columns],
        separator=op.separator
    ).alias(op.output_column)
    return df.with_columns(concat_expr)
