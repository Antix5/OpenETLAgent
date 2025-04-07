import polars as pl
from app.models import ArithmeticOperation

def apply_arithmetic(df: pl.DataFrame, op: ArithmeticOperation) -> pl.DataFrame:
    """Applies the arithmetic operation."""
    if len(op.input_columns) != 2:
        raise ValueError("ArithmeticOperation requires exactly two input columns.")

    col_a_name, col_b_name = op.input_columns
    if col_a_name not in df.columns:
        raise ValueError(f"ArithmeticOperation: Input column '{col_a_name}' not found in DataFrame columns: {df.columns}")
    if col_b_name not in df.columns:
        raise ValueError(f"ArithmeticOperation: Input column '{col_b_name}' not found in DataFrame columns: {df.columns}")

    col_a, col_b = pl.col(col_a_name), pl.col(col_b_name)

    if op.operator == '+':
        arithmetic_expr = (col_a + col_b)
    elif op.operator == '-':
        arithmetic_expr = (col_a - col_b)
    elif op.operator == '*':
        arithmetic_expr = (col_a * col_b)
    elif op.operator == '/':
        arithmetic_expr = (col_a / col_b)
    else:
        raise ValueError(f"ArithmeticOperation: Unsupported operator '{op.operator}'")

    return df.with_columns(arithmetic_expr.alias(op.output_column))
