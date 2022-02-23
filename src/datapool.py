from typing import List
from .video import Video
from tqdm.auto import tqdm

class DataPool(dict):
    def __init__(self, files: List[str], config):
        super(DataPool, self).__init__()
        for file in tqdm(files, desc='Video loading'):
            self.__setitem__(file, Video(file, config))

    def __getitem__(self, key) -> Video:
        return super().get(key)

    def __iter__(self) -> Video:
        for v in super().values():
            yield v
