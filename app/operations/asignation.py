# In app/operations/assignation.py
import polars as pl
import logging
from app.models import AssignationOperation # Assuming this is your model import

def apply_assignation(df: pl.DataFrame, op: AssignationOperation) -> pl.DataFrame:
    """
    Applies the assignation operation, assigning either a literal value
    or copying from another column specified in op.value.
    """
    value_to_assign = None

    # Check if op.value is a string AND if that string is an existing column name
    if isinstance(op.value, str) and op.value in df.columns:
        # Assign values FROM the specified source column
        value_to_assign = pl.col(op.value)
        logging.debug(f"AssignationOperation: Copying from column '{op.value}' to '{op.output_column}'")
    else:
        # Assign the value itself as a LITERAL
        value_to_assign = pl.lit(op.value)
        logging.debug(f"AssignationOperation: Assigning literal value '{op.value}' (type: {type(op.value)}) to '{op.output_column}'")

    try:
        return df.with_columns(value_to_assign.alias(op.output_column))
    except Exception as e:
        raise ValueError(f"Failed to apply assignation to create column '{op.output_column}' with value '{op.value}': {e}") from e