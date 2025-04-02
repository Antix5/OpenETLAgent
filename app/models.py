from pydantic import BaseModel, Field, PositiveInt # Re-add PositiveInt if needed elsewhere, or remove if not
from typing import List, Any, Literal, Union, Dict, Annotated

# --- Schema Definition ---

class ColumnDefinition(BaseModel):
    name: str
    type: Literal['string', 'integer', 'float', 'boolean', 'positive integer'] # Extend as needed
    description: str | None = None

class FileSchema(BaseModel):
    name: str
    # input_format: str | None = None # Maybe needed for input schema
    output_format: Literal['csv', 'json', 'parquet'] | None = None # For output schema
    columns: Dict[str, ColumnDefinition] # Use column technical name as key

# --- Base Operation ---

class BaseOperation(BaseModel):
    # Revert to regular field for Pydantic discriminator
    operation_type: str = Field(..., description="Discriminator field for the operation type")
    output_column: str = Field(..., description="Name of the column to create/modify")

# --- Specific Operations ---

class EqualityOperation(BaseOperation):
    operation_type: Literal['equality'] = 'equality' # type: ignore
    input_column: str

class ConcatenationOperation(BaseOperation):
    operation_type: Literal['concatenation'] = 'concatenation' # type: ignore
    input_columns: List[str]
    separator: str = "" # Optional separator

class ApplicationOperation(BaseOperation):
    operation_type: Literal['application'] = 'application' # type: ignore
    input_columns: List[str]
    # How to represent the function?
    # Option 1: String containing Python lambda expression (Security Risk!)
    function_str: str = Field(..., description="Python lambda expression string, e.g., 'lambda r: r['col_a'] + r['col_b']'. 'r' represents the row.")
    # Option 2: Reference to a predefined function name
    # function_name: str

class SwitchingOperation(BaseOperation):
    operation_type: Literal['switching'] = 'switching' # type: ignore
    condition_column: str # Assumed to be boolean type
    true_column: str # Column to use if condition is true
    false_column: str # Column to use if condition is false

class AssignationOperation(BaseOperation):
    operation_type: Literal['assignation'] = 'assignation' # type: ignore
    value: Any # The constant value to assign

class CastingOperation(BaseOperation):
    operation_type: Literal['casting'] = 'casting' # type: ignore
    input_column: str
    target_type: Literal['string', 'integer', 'float', 'boolean', 'positive integer'] # Match ColumnDefinition types

# --- Pipeline ---

# Union of all possible operation types using discriminated union
AnyOperation = Annotated[
    Union[
        EqualityOperation,
        ConcatenationOperation,
        ApplicationOperation,
        SwitchingOperation,
        AssignationOperation,
        CastingOperation,
    ],
    Field(discriminator='operation_type')
]


class EtlPipeline(BaseModel):
    input_schema: FileSchema
    output_schema: FileSchema
    operations: List[AnyOperation] # The sequence of operations
