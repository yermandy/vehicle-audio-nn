import numpy as np
import src 

class Video():
    def __init__(self, file: str, window_length: float = 6, split_ratio: float = 25 * 60):
        print(f'loading: {file}')
        self.file = file
        self.signal, self.sr = src.load_audio(f'data/audio/{file}.MP4.wav', return_sr=True)
        self.events = src.load_events(f'data/labels/{file}.MP4.txt')
        self.intervals, self.events_in_interval = src.load_intervals(f'data/intervals/{file}.MP4.txt')
        self.signal_length = len(self.signal) / self.sr
        self._split(window_length, split_ratio)

    def _split(self, window_length: float, split_ratio: float):
        split_at = self.signal_length * split_ratio
        idx = np.abs(self.intervals - split_at).argmin()
        split_time = self.intervals[idx]
        split_time = np.round(split_time / window_length) * window_length

        self.trn_from_time = 0
        self.trn_till_time = split_time
        self.val_from_time = split_time
        self.val_till_time = (self.signal_length //window_length) * window_length

        print(f' --> trn time {self.trn_from_time} : {self.trn_till_time}\n --> val time {self.val_from_time} : {self.val_till_time}')

    def get_from_till_time(self, is_trn):
        if is_trn:
            return self.trn_from_time, self.trn_till_time
        else:
            return self.val_from_time, self.val_till_time

    def get_events_count(self, is_trn):
        events = self.events
        if is_trn:
            mask = (events >= self.trn_from_time) & (events < self.trn_till_time)
        else:
            mask = (events >= self.val_from_time) & (events < self.val_till_time)
        return mask.sum()
