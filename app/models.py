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

class ArithmeticOperation(BaseOperation):
    operation_type: Literal['arithmetic'] = 'arithmetic' # type: ignore
    input_columns: List[str] = Field(..., min_length=2, max_length=2, description="Exactly two input columns for the operation.")
    operator: Literal['+', '-', '*', '/'] = Field(..., description="The arithmetic operator to apply.")

class ComparisonOperation(BaseOperation):
    operation_type: Literal['comparison'] = 'comparison' # type: ignore
    input_column: str = Field(..., description="The input column to compare.")
    operator: Literal['==', '!=', '>', '<', '>=', '<='] = Field(..., description="The comparison operator.")
    # Value can be string, int, float. Pydantic handles validation.
    # We'll cast the column to the value's type or vice-versa if needed during application.
    value: Any = Field(..., description="The value to compare against.")

class BindOperation(BaseOperation):
    # Note: output_column is inherited but not strictly needed here, as columns are added based on 'columns_to_add'
    operation_type: Literal['bind'] = 'bind' # type: ignore
    right_file_path: str = Field(..., description="Path to the CSV file to join with (e.g., 'input_folder/customers.csv').")
    # Provide schema as dict {col_name: type_string} e.g. {"cust_id": "integer", "full_name": "string"}
    right_schema_columns: Dict[str, str] = Field(..., description="Schema of the right file as a dictionary {column_name: type_string}.")
    left_on: str = Field(..., description="Column name from the left (current) DataFrame to join on.")
    right_on: str = Field(..., description="Column name from the right DataFrame to join on.")
    how: Literal['left', 'inner', 'outer', 'cross'] = Field(default='left', description="Type of join to perform.")
    columns_to_add: List[str] = Field(..., description="List of column names from the right DataFrame to add to the left DataFrame.")


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
        ArithmeticOperation,
        ComparisonOperation,
        BindOperation, # Added new operation type
    ],
    Field(discriminator='operation_type')
]


class EtlPipeline(BaseModel):
    input_schema: FileSchema
    output_schema: FileSchema
    operations: List[AnyOperation] # The sequence of operations
