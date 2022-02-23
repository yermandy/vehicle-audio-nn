import numpy as np
import src 
from .constants import * 


class Video():
    def __init__(self, file: str, config, silent: bool = True):
        if not silent:
            print(f'loading: {file}')
            
        self.silent = silent
        self.file = file
        self.signal, self.sr = src.load_audio(file, resample_sr=config.sr, return_sr=True)
        self.events = src.load_events(file)
        self.intervals = src.load_intervals(file)[:, 1]
        self.signal_length = len(self.signal) / self.sr
        self._split(config.window_length, config.split_ratio)

    def _split(self, window_length: float, split_ratio: float):
        split_at = self.signal_length * split_ratio
        idx = np.abs(self.intervals - split_at).argmin()
        split_time = self.intervals[idx]
        split_time = np.round(split_time / window_length) * window_length

        self.trn_from_time = 0
        self.trn_till_time = split_time
        self.val_from_time = split_time
        self.val_till_time = (self.signal_length // window_length) * window_length

        if not self.silent:
            print(f' --> trn time {self.trn_from_time} : {self.trn_till_time}\n --> val time {self.val_from_time} : {self.val_till_time}')

    def get_events(self, part: Part):
        if part.is_trn():
            return src.crop_events(self.events, self.trn_from_time, self.trn_till_time)
        elif part.is_val():
            return src.crop_events(self.events, self.val_from_time, self.val_till_time)
        else:
            return self.events

    def get_signal(self, part: Part):
        if part.is_trn():
            return src.crop_signal(self.signal, self.sr, self.trn_from_time, self.trn_till_time)
        elif part.is_val():
            return src.crop_signal(self.signal, self.sr, self.val_from_time, self.val_till_time)
        else:
            return self.signal

    def get_from_till_time(self, part: Part):
        if part.is_trn():
            return self.trn_from_time, self.trn_till_time
        elif part.is_val():
            return self.val_from_time, self.val_till_time
        else:
            return self.trn_from_time, self.val_till_time

    def get_events_count(self, part: Part):
        events = self.events
        if part.is_trn():
            return  np.sum((events >= self.trn_from_time) & (events < self.trn_till_time))
        elif part.is_val():
            return np.sum((events >= self.val_from_time) & (events < self.val_till_time))
        else:
            return len(events)

    def __str__(self) -> str:
        return f'{self.file} ({int(self.trn_from_time)}:{int(self.trn_till_time)}) ({int(self.val_from_time)}:{int(self.val_till_time)})'