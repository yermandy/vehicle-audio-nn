from enum import Enum

class Normalization(str, Enum):
    NONE = "none"
    ROW_WISE = "row-wise"
    COLUMN_WISE = "column-wise"
    GLOBAL = 'global'

    def __str__(self) -> str:
        return str.__str__(self)
