from enum import Enum


class Normalization(str, Enum):
    NONE = "none"
    ROW_WISE = "row-wise"
    COLUMN_WISE = "column-wise"
    GLOBAL = 'global'

    def __str__(self) -> str:
        return str.__str__(self)

    def is_global(self) -> bool:
        return self == self.GLOBAL

    def is_row_wise(self) -> bool:
        return self == self.ROW_WISE

    def is_column_wise(self) -> bool:
        return self == self.COLUMN_WISE

    def is_none(self) -> bool:
        return self == self.NONE


class Part(str, Enum):
    TRAINING = 'trn'
    VALIDATION = 'val'
    TEST = 'tst'

    def is_trn(self) -> bool:
        return self == self.TRAINING

    def is_val(self) -> bool:
        return self == self.VALIDATION

    def is_tst(self) -> bool:
        return self == self.TEST

    def __str__(self) -> str:
        return str.__str__(self)