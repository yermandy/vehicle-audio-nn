from typing import List
from .video import Video


class DataPool(dict):
    def __init__(self,
                 files: List[str],
                 window_length: float = 6,
                 split_ratio: float = 25 * 60):
        super(DataPool, self).__init__()
        for file in files:
            self.__setitem__(file, Video(file, window_length, split_ratio))

    def __getitem__(self, key) -> Video:
        return super().get(key)

    def __iter__(self) -> Video:
        for v in super().values():
            yield v
