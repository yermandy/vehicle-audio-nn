import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from scipy.stats import norm

class VehicleDataset(Dataset):

    def __init__(self,
                 audio_file,
                 labels_file,
                 start_time=0,
                 end_time=int(1e8),
                 window_len=1.0,
                 seed=np.random.randint(0, int(1e8)),
                 sampling='random',
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

        print(f'all: {len(self.labels)} | positive: {np.sum(self.labels)}')

        self.n_fft = 1024
        self.hop_length = 128
        self.n_mels = 64

        melkwargs = {
            "n_fft" : self.n_fft, 
            "n_mels" : self.n_mels, 
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

        N = np.ceil((end_time - start_time) / window_len).astype(int)

        for i in range(len(signals)):
            x = np.linspace(i * sr, (i + 1) * sr, 100)
            # y = 0
            # for e in events:
            y = np.sum([norm.pdf(x, e, 3000) for e in events], axis=0)
            # y_exp = np.exp(y)
            # y = y_exp / y_exp.sum()
            labels.append(y)


        labels = np.array(labels, dtype=np.float32)
        signals = np.array(signals)

        # sampler = RandomUnderSampler(random_state=self.seed)
        # signals, labels = sampler.fit_resample(signals, labels)

        self.labels = labels
        self.signals = torch.tensor(signals)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index, use_librosa=True):
        signal = self.signals[index]

        mel_features = self.mel_transform(signal).unsqueeze(0)
        mfcc_features = self.mfcc_transform(signal).unsqueeze(0)
        features = torch.cat((mel_features, mfcc_features), dim=1)

        return features, self.labels[index]