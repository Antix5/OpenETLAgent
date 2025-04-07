import polars as pl
import logging
from app.models import ApplicationOperation

def apply_application(df: pl.DataFrame, op: ApplicationOperation) -> pl.DataFrame:
    """Applies a user-defined function string to specified columns."""
    missing_inputs = [col for col in op.input_columns if col not in df.columns]
    if missing_inputs:
        raise ValueError(f"ApplicationOperation: Input columns not found: {missing_inputs} in DataFrame columns: {df.columns}")

    logging.warning(f"Executing ApplicationOperation '{op.output_column}' using potentially slow map_elements and eval(). Ensure 'function_str' is trusted.")
    try:
        # Define safe builtins for eval context
        safe_builtins = {
            "float": float, "int": int, "str": str, "list": list, "dict": dict,
            "set": set, "tuple": tuple, "True": True, "False": False, "None": None
        }
        # Evaluate the function string in a restricted environment
        lambda_func = eval(op.function_str, {"__builtins__": safe_builtins}, {})

        # Apply the function using map_elements
        map_expr = pl.struct(op.input_columns).map_elements(
            lambda row_struct: lambda_func(row_struct),
            # Defaulting to string, casting might be needed later based on schema
            return_dtype=pl.Utf8
        ).alias(op.output_column)

        return df.with_columns(map_expr)
    except Exception as e:
        raise ValueError(f"Error executing function_str '{op.function_str}' for ApplicationOperation '{op.output_column}': {e}")
