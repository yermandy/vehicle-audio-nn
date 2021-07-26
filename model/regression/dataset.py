import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset


class VehicleDataset(Dataset):

    def __init__(self,
                 audio_file,
                 labels_file,
                 start_time=0,
                 end_time=int(1e8),
                 window_len=3.0,
                 seed=np.random.randint(0, int(1e8)),
                 use_offset=False):

        # window_len, start_time, end_time in seconds
        self.window_len = window_len
        self.start_time = start_time
        self.end_time = end_time
        self.use_offset = use_offset
        self.seed = seed

        # load and convert to mono
        self.signal, self.sr = torchaudio.load(audio_file)
        self.signal = self.signal.mean(0)
        # signal = signal.numpy()

        # load labels
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
        events = self.events.copy()

        window_len = self.window_len
        start_time = self.start_time
        end_time = self.end_time

        sr = self.sr

        # crop signal
        offset = np.random.uniform(0, window_len, 1)[0] if self.use_offset else 0
        signal = signal[int((start_time + offset) * sr): end_time * sr + 1]

        n_samples_per_window = int(sr * window_len)

        # split signal to windows of size window_len [sec]
        signals = signal.split(n_samples_per_window)

        # remove last window
        signals = signals[:-1]
        signals = np.array([s.numpy() for s in signals])

        # load events in seconds, crop and convert to events in samples
        events = events - start_time + offset
        events = events[events < end_time - start_time]
        events *= sr

        labels = []
        for i in range(len(signals)):
            counts = (i * n_samples_per_window <= events) & \
                (events < i * n_samples_per_window + n_samples_per_window)
            counts = counts.sum()
            labels.append(counts)

        labels = np.array(labels)
        signals = np.array(signals)

        mask = labels != 0

        pos_labels = labels[mask]
        pos_signals = signals[mask]

        neg_labels = labels[~mask]
        neg_signals = signals[~mask]

        np.random.seed(self.seed)

        neg_random_idx = np.random.choice(
            range(len(neg_signals)), len(pos_labels))
        neg_labels = neg_labels[neg_random_idx]
        neg_signals = neg_signals[neg_random_idx]

        labels = np.concatenate((pos_labels, neg_labels))
        signals = np.concatenate((pos_signals, neg_signals))

        self.labels = labels
        self.signals = torch.tensor(signals)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        signal = self.signals[index]
        label = self.labels[index]
        features = self.transform(signal)
        return features, label