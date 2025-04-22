import polars as pl
import logging
from app.models import SwitchingOperation

def apply_switching(df: pl.DataFrame, op: SwitchingOperation) -> pl.DataFrame:
    """
    Applies a multi-way switching operation based on values in a switch column.
    Copies data from a source column specified in the mapping based on the
    value found in the switch_column for that row. Uses a default if no match.
    """
    if not op.output_column:
        raise ValueError("SwitchingOperation requires an 'output_column'.")
    if op.switch_column not in df.columns:
        raise ValueError(f"SwitchingOperation: Switch column '{op.switch_column}' not found in DataFrame columns: {df.columns}")

    # Validate source columns in mapping exist
    for source_col in op.mapping.values():
        if source_col not in df.columns:
            raise ValueError(f"SwitchingOperation: Source column '{source_col}' from mapping not found in DataFrame columns: {df.columns}")

    # Validate default column if provided
    if op.default_column and op.default_column not in df.columns:
         raise ValueError(f"SwitchingOperation: Default column '{op.default_column}' not found in DataFrame columns: {df.columns}")

    # Build the nested when/then expression chain
    switch_expr = None
    for switch_value, source_column_name in op.mapping.items():
        condition = (pl.col(op.switch_column) == pl.lit(switch_value))
        then_expr = pl.col(source_column_name)

        if switch_expr is None:
            # Start the chain
            switch_expr = pl.when(condition).then(then_expr)
        else:
            # Continue the chain
            switch_expr = switch_expr.when(condition).then(then_expr)

    # Add the mandatory default case
    if switch_expr is None:
        # Handle edge case where mapping is empty (shouldn't happen with validation but good practice)
        logging.warning("SwitchingOperation: Mapping is empty. Applying default directly.")
        if op.default_column:
            default_expr = pl.col(op.default_column)
        else: # default_value must be set due to model validation
            default_expr = pl.lit(op.default_value)
        switch_expr = default_expr.alias(op.output_column)
    else:
        if op.default_column:
            default_expr = pl.col(op.default_column)
        else: # default_value must be set
            default_expr = pl.lit(op.default_value)
        switch_expr = switch_expr.otherwise(default_expr).alias(op.output_column)

    logging.debug(f"Applying switching: switch_column='{op.switch_column}', mapping={op.mapping}, default_column='{op.default_column}', default_value='{op.default_value}', output_column='{op.output_column}'")

    try:
        return df.with_columns(switch_expr)
    except Exception as e:
        # Provide more context in error
        raise ValueError(f"Failed to apply switching operation for output column '{op.output_column}': {e}") from e
