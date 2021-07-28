import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from ..utils import *


class VehicleDataset(Dataset):

    def __init__(self,
                 audio_file,
                 labels_file,
                 start_time=0,
                 end_time=int(1e8),
                 frame_length=3.0,
                 seed=np.random.randint(0, int(1e8)),
                 use_offset=False):

        self.frame_length = frame_length
        self.start_time = start_time
        self.end_time = end_time
        self.use_offset = use_offset
        self.seed = seed

        self.signal, self.sr = torchaudio.load(audio_file)
        self.signal = self.signal.mean(0)

        self.events = np.loadtxt(labels_file)

        self.split_signal()

        print(f'all: {len(self.labels)} | positive: {sum(self.labels)}')

        self.n_fft = 1024
        self.hop_length = 128
        self.n_mels = 64

        melkwargs = {
            "n_fft": self.n_fft,
            "n_mels": self.n_mels,
            "hop_length": self.hop_length
        }

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            **melkwargs
        )

        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sr,
            n_mfcc=8,
            melkwargs=melkwargs
        )

        self.transform = lambda signal: torch.cat((self.mel_transform(signal), self.mfcc_transform(signal)), dim=0).unsqueeze(0)

    def split_signal(self):
        signal = self.signal
        events = self.events

        frame_length = self.frame_length
        start_time = self.start_time
        end_time = self.end_time

        sr = self.sr

        n_samples_per_frame = int(sr * frame_length)

        offset = get_offset(frame_length) if self.use_offset else 0

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
