from typing import Tuple
import numpy as np
import torch
import src
from src.config import Config
from .constants import *


class Video:
    def __init__(
        self, file: str, config: Config, silent: bool = True, csv_version: int = 0
    ):
        if not silent:
            print(f"loading: {file}")
        self.silent = silent
        if isinstance(file, list):
            file, csv_version = file[0], file[1]
        self.file = file
        self.config = config
        self.domain = 0
        self.signal, self.sr = src.load_audio(
            file,
            resample_sr=config.sr,
            return_sr=True,
            normalize=config.signal_normalization,
        )
        self.manual_counts = src.load_manual_counts(file)
        self.events = src.load_events(file)

        self.csv = src.load_csv(file, csv_version)
        self.views = src.load_views_from_csv(self.csv, csv_version)
        self.category = src.load_category_from_csv(self.csv, csv_version)
        self.events_start_time, self.events_end_time = src.load_event_time_from_csv(
            self.csv, csv_version
        )
        self.intervals = src.load_intervals(file)
        self.signal_length = len(self.signal) / self.sr
        self._split(config.window_length, config.split_ratio)

    def _split(self, window_length: float, split_ratio: float):
        if len(self.intervals) == 1:
            split_time = self.intervals[0, 1]
        elif len(self.intervals) > 1:
            split_at = self.signal_length * split_ratio
            idx = np.abs(self.intervals[:-1, 1] - split_at).argmin()
            split_time = self.intervals[idx, 1]
            split_time = np.round(split_time / window_length) * window_length
        else:
            split_time = 0

        self.left_from_time = 0
        self.left_till_time = split_time
        self.right_from_time = split_time
        self.right_till_time = (self.signal_length // window_length) * window_length

        if not self.silent:
            print(
                f" --> trn time {self.left_from_time} : {self.left_till_time}\n --> val time {self.right_from_time} : {self.right_till_time}"
            )

    def get_events(self, part=Part.WHOLE) -> np.ndarray:
        if part.is_left():
            return src.crop_events(
                self.events, self.left_from_time, self.left_till_time
            )
        elif part.is_right():
            return src.crop_events(
                self.events, self.right_from_time, self.right_till_time
            )
        else:
            return src.crop_events(
                self.events, self.left_from_time, self.right_till_time
            )

    def get_signal(self, part=Part.WHOLE) -> torch.Tensor:
        if part.is_left():
            return src.crop_signal(
                self.signal, self.sr, self.left_from_time, self.left_till_time
            )
        elif part.is_right():
            return src.crop_signal(
                self.signal, self.sr, self.right_from_time, self.right_till_time
            )
        else:
            return src.crop_signal(
                self.signal, self.sr, self.left_from_time, self.right_till_time
            )

    def get_from_till_time(self, part=Part.WHOLE) -> Tuple[float, float]:
        if part.is_left():
            return self.left_from_time, self.left_till_time
        elif part.is_right():
            return self.right_from_time, self.right_till_time
        else:
            return self.left_from_time, self.right_till_time

    def get_events_count(self, part=Part.WHOLE) -> int:
        events = self.events
        if part.is_left():
            return np.sum(
                (events >= self.left_from_time) & (events < self.left_till_time)
            )
        elif part.is_right():
            return np.sum(
                (events >= self.right_from_time) & (events < self.right_till_time)
            )
        else:
            return np.sum(
                (events >= self.left_from_time) & (events < self.right_till_time)
            )

    def __str__(self) -> str:
        return f"{self.file} ({int(self.left_from_time)}:{int(self.left_till_time)}) ({int(self.right_from_time)}:{int(self.right_till_time)})"
