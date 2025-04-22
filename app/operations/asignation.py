# In app/operations/assignation.py
import polars as pl
import logging
from app.models import AssignationOperation # Assuming this is your model import

def apply_assignation(df: pl.DataFrame, op: AssignationOperation) -> pl.DataFrame:
    """
    Applies the assignation operation, assigning a literal value
    specified in op.value to the output column.
    """
    if not op.output_column:
        raise ValueError("AssignationOperation requires an 'output_column'.")

    # Assign the value itself as a LITERAL
    value_to_assign = pl.lit(op.value)
    logging.debug(f"AssignationOperation: Assigning literal value '{op.value}' (type: {type(op.value)}) to '{op.output_column}'")

    try:
        return df.with_columns(value_to_assign.alias(op.output_column))
    except Exception as e:
        # Consider more specific error handling if needed
        raise ValueError(f"Failed to apply assignation to create column '{op.output_column}' with literal value '{op.value}': {e}") from e
