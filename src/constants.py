from enum import Enum, IntEnum


class CsvColumnID(IntEnum):
    START_TIME = 8
    END_TIME = 9
    BEST_DETECTION_FRAME_TIME = 14
    CATEGORY = 15
    COLOR = 17
    VIEWS = 23


class InferenceFunction(str, Enum):
    SIMPLE = 'simple'
    OPTIMAL_RVCE = 'optimal_rvce'
    DOUBLED = 'doubled'

    def __str__(self) -> str:
        return str.__str__(self)

    def is_simple(self) -> bool:
        return self == self.SIMPLE

    def is_optimal_rvce(self) -> bool:
        return self == self.OPTIMAL_RVCE

    def is_doubled(self) -> bool:
        return self == self.DOUBLED


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


class Transformation(str, Enum):
    STFT = "stft"
    MEL = "mel"
    MFCC = "mfcc"

    def __str__(self) -> str:
        return str.__str__(self)

    def is_stft(self) -> bool:
        return self == self.STFT

    def is_mel(self) -> bool:
        return self == self.MEL

    def is_mfcc(self) -> bool:
        return self == self.MFCC


class Part(str, Enum):
    LEFT = 'left'
    RIGHT = 'right'
    WHOLE = 'whole'

    def __str__(self) -> str:
        return str.__str__(self)

    def is_left(self) -> bool:
        return self == self.LEFT

    def is_right(self) -> bool:
        return self == self.RIGHT

    def is_whole(self) -> bool:
        return self == self.WHOLE