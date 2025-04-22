from pydantic import BaseModel, Field, model_validator
from typing import List, Any, Literal, Union, Dict, Annotated, Optional, Sequence
from polars import Utf8, Int64, Float64, Boolean, Date # Added Date

# --- Type Mapping ---
POLARS_TYPE_MAP = {
    'string': Utf8,
    'integer': Int64,
    'float': Float64,
    'boolean': Boolean,
    'date': Date, # Added Date mapping
    'positive integer': Int64 # Keep positive integer mapping for schema definition
}
SCHEMA_TYPE_MAP_REVERSE = {v: k for k, v in POLARS_TYPE_MAP.items()}
# Add reverse mapping for Date if needed
SCHEMA_TYPE_MAP_REVERSE[Date] = 'date'


# --- Schema Definition ---

class ColumnDefinition(BaseModel):
    # Removed name field, as it will be the key in the dictionary
    type: Literal['string', 'integer', 'float', 'boolean', 'date', 'positive integer'] # Kept 'positive integer'
    description: str | None = None

class FileSchema(BaseModel):
    name: str
    columns: Dict[str, ColumnDefinition] # Use column technical name as key

# --- Input/Output Definitions ---

class FileDefinition(BaseModel):
    path: str = Field(..., description="Path to the file.")
    format: Literal['csv', 'json', 'parquet'] = Field(default='csv', description="Format for the file.")
    file_schema: FileSchema = Field(..., description="Schema definition of the file.")


# --- Base Operation ---

class BaseOperation(BaseModel):
    operation_type: str = Field(..., description="Discriminator field for the operation type")
    output_column: Optional[str] = Field(default=None, description="Name of the primary column to create/modify (if applicable).")

# --- Specific Operations (Keep all existing operation models) ---

class EqualityOperation(BaseOperation):
    operation_type: Literal['equality'] = 'equality' # type: ignore
    input_column: str

class ConcatenationOperation(BaseOperation):
    operation_type: Literal['concatenation'] = 'concatenation' # type: ignore
    input_columns: List[str]
    separator: str = ""

class ApplicationOperation(BaseOperation):
    operation_type: Literal['application'] = 'application' # type: ignore
    input_columns: List[str]
    function_str: str = Field(..., description="Python lambda expression string, e.g., 'lambda r: r['col_a'] + r['col_b']'. 'r' represents the row.")

class SwitchingOperation(BaseOperation):
    operation_type: Literal['switching'] = 'switching' # type: ignore
    switch_column: str = Field(..., description="Column whose values determine the switch.")
    mapping: Dict[Any, str] = Field(..., description="Dictionary mapping values in switch_column to source column names to copy from.")
    default_column: Optional[str] = Field(default=None, description="Column to copy from if no mapping key matches. Mutually exclusive with default_value.")
    default_value: Optional[Any] = Field(default=None, description="Literal value to assign if no mapping key matches. Mutually exclusive with default_column.")

    @model_validator(mode='before')
    @classmethod
    def check_default_exclusive(cls, values):
        default_column = values.get('default_column')
        default_value = values.get('default_value')
        if default_column is not None and default_value is not None:
            raise ValueError("Provide either 'default_column' or 'default_value', not both.")
        if default_column is None and default_value is None:
            raise ValueError("Must provide either 'default_column' or 'default_value'.")
        return values

class AssignationOperation(BaseOperation):
    operation_type: Literal['assignation'] = 'assignation' # type: ignore
    value: Any

class CastingOperation(BaseOperation):
    operation_type: Literal['casting'] = 'casting' # type: ignore
    input_column: str
    target_type: Literal['string', 'integer', 'float', 'boolean', 'date', 'positive integer'] # Kept 'positive integer'

class ArithmeticOperation(BaseOperation):
    operation_type: Literal['arithmetic'] = 'arithmetic' # type: ignore
    input_columns: List[str] = Field(..., min_length=2, max_length=2, description="Exactly two input columns for the operation.")
    operator: Literal['+', '-', '*', '/']

class ComparisonOperation(BaseOperation):
    operation_type: Literal['comparison'] = 'comparison' # type: ignore
    input_column: str
    operator: Literal['==', '!=', '>', '<', '>=', '<=']
    value: Any

class BindOperation(BaseOperation):
    operation_type: Literal['bind'] = 'bind' # type: ignore
    right_file_path: str = Field(..., description="Path or input key (e.g., 'customers_input') to the file to join with.")
    right_schema_columns: Dict[str, str] = Field(..., description="Schema of the right file as a dictionary {column_name: type_string}.")
    left_on: str
    right_on: str
    how: Literal['left', 'inner', 'outer', 'cross'] = Field(default='left')
    columns_to_add: List[str]

class FoldOperation(BaseOperation):
    operation_type: Literal['fold'] = 'fold' # type: ignore
    id_columns: List[str]
    value_columns: List[str]
    key_column_name: str
    value_column_name: str

class UnfoldOperation(BaseOperation):
    operation_type: Literal['unfold'] = 'unfold' # type: ignore
    index_columns: List[str]
    key_column: str
    value_column: str

# --- Union of Operations ---

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
        BindOperation,
        FoldOperation,
        UnfoldOperation,
    ],
    Field(discriminator='operation_type')
]

# --- New Configuration and Flow Models ---

class PipelineConfig(BaseModel):
    """Defines the overall configuration including inputs and outputs."""
    inputs: Dict[str, FileDefinition] = Field(..., description="Dictionary of named inputs, each with a path and schema.")
    outputs: Dict[str, FileDefinition] = Field(..., description="Dictionary of named outputs, each with a path, format, and schema.")

class PipelineFlow(BaseModel):
    """Defines the source and operations for a single pipeline flow targeting one output."""
    source: str = Field(..., description="Key referencing an input defined in the main config.")
    operations: List[AnyOperation] = Field(..., description="The sequence of operations for this specific flow.")
