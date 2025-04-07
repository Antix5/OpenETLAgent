import polars as pl
from app.models import AssignationOperation

def apply_assignation(df: pl.DataFrame, op: AssignationOperation) -> pl.DataFrame:
    """Applies the assignation operation (assigns a literal value)."""
    # Note: Handling of temporary columns (like '_*_literal') remains in the main apply_operations loop
    return df.with_columns(
        pl.lit(op.value).alias(op.output_column)
    )
