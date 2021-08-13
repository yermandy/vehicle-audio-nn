import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from ..utils import *
from copy import deepcopy

from easydict import EasyDict

class VehicleDataset(Dataset):

    def __init__(self,
            signal: torch.Tensor,
            events: np.array,
            start_time: float = 0,
            end_time: float = int(1e8),
            seed: int = np.random.randint(0, int(1e8)),
            use_offset: bool = False,
            params: EasyDict = EasyDict()):

        self.start_time = start_time
        self.end_time = end_time
        self.use_offset = use_offset
        self.seed = seed
        self.params = deepcopy(params)

        self.signal = deepcopy(signal)
        self.events = deepcopy(events)
        self.window_length = params.nn_hop_length * (params.n_frames - 1) + params.frame_length

        self.split_signal()

        # print(f'all: {len(self.labels)} | positive: {sum(self.labels)}')

        melkwargs = {
            "n_fft": params.n_fft,
            "n_mels": params.n_mels,
            "hop_length": params.hop_length
        }

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=params.sr,
            **melkwargs
        )

        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=params.sr,
            n_mfcc=params.n_mfcc,
            melkwargs=melkwargs
        )

        self.transform = lambda signal: torch.cat((self.mel_transform(signal), self.mfcc_transform(signal)), dim=0).unsqueeze(0)
    
    def split_signal(self):
        signal = self.signal
        events = self.events

        window_length = self.window_length
        start_time = self.start_time
        end_time = self.end_time

        sr = self.params.sr

        n_samples_per_frame = int(sr * window_length)

        offset = get_offset(window_length) if self.use_offset else 0

        signal = transform_signal(signal, start_time, end_time, offset, sr)

        signals = split_signal(signal, n_samples_per_frame)

        events = transform_events(events, start_time, end_time, offset, sr)

        labels = get_counts_labels(events, len(signals), n_samples_per_frame)

        signals, labels = under_sampling(signals, labels, self.seed)

        self.labels = labels
        self.signals = torch.tensor(signals)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        signal = self.signals[index]
        label = self.labels[index]
        features = self.transform(signal)
        return features, label


class VehicleValidationDataset(Dataset):
    
    def __init__(self, signal: torch.Tensor, params: EasyDict, transform):
        self.signal = deepcopy(signal)
        self.params = deepcopy(params)
        self.transform = transform
        
    def __len__(self):
        return self.params.n_hops

    def __getitem__(self, index):
        start = index * self.params.n_samples_in_nn_hop
        end = start + self.params.n_samples_in_frame
        x = self.params.signal[start: end]
        x = self.transform(x)
        return x