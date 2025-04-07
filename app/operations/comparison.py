# In app/operations/comparison.py
import polars as pl
import logging
from app.models import ComparisonOperation

def apply_comparison(df: pl.DataFrame, op: ComparisonOperation) -> pl.DataFrame:
    """Applies the comparison operation using structural pattern matching."""
    if op.input_column not in df.columns:
        raise ValueError(f"ComparisonOperation: Input column '{op.input_column}' not found in DataFrame columns: {df.columns}")

    # Use the input column directly without forcing a cast
    input_col_expr = pl.col(op.input_column)
    # Create a Polars literal from the value provided in the operation
    comparison_value_lit = pl.lit(op.value)

    # Get current dtype for logging/debugging if needed
    current_dtype = df[op.input_column].dtype
    logging.debug(f"ComparisonOperation: Comparing column '{op.input_column}' (dtype: {current_dtype}) with literal '{op.value}' (type: {type(op.value)})")

    # Use match/case to build the comparison expression
    match op.operator:
        case '==':
            comparison_expr = (input_col_expr == comparison_value_lit)
        case '!=':
            comparison_expr = (input_col_expr != comparison_value_lit)
        case '>':
            comparison_expr = (input_col_expr > comparison_value_lit)
        case '<':
            comparison_expr = (input_col_expr < comparison_value_lit)
        case '>=':
            comparison_expr = (input_col_expr >= comparison_value_lit)
        case '<=':
            comparison_expr = (input_col_expr <= comparison_value_lit)
        case _: # Default case for unsupported operators
            raise ValueError(f"ComparisonOperation: Unsupported operator '{op.operator}'")

    try:
        # Apply the comparison expression
        return df.with_columns(comparison_expr.alias(op.output_column))
    except Exception as e:
        # Catch potential errors during comparison application (e.g., truly incompatible types for >, <)
        raise ValueError(f"Failed to apply comparison '{op.operator}' between column '{op.input_column}' (dtype: {current_dtype}) and value '{op.value}' (type: {type(op.value)}): {e}") from e