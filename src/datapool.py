from typing import List
from .video import Video
import src
from tqdm.auto import tqdm
from tqdm.rich import tqdm_rich


class DataPool(dict):
    def __init__(self, files: List[str], config):
        super(DataPool, self).__init__()
        for file in tqdm_rich(files, desc="Video loading"):
            video = Video(file, config)
            # TODO change this constant to something better
            if src.get_signal_length(video.signal, config) < 10:
                continue
            self.__setitem__(file[0] if isinstance(file, list) else file, video)

    def __getitem__(self, key) -> Video:
        return super().get(key)

    def __iter__(self) -> Video:
        for v in super().values():
            yield v
