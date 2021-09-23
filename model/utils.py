import torch
import numpy as np
import random
import torchaudio


def get_offset(max_offset: float):
    return random.uniform(0, max_offset)


def get_window_length(params):
    return params.nn_hop_length * (params.n_frames - 1) + params.frame_length

def transform_signal(signal: torch.Tensor, start_time: float, end_time: float, offset: float, sr: int):
    return signal[int((start_time + offset) * sr): int(end_time * sr)]


def transform_events(events: np.array, start_time: float, end_time: float, offset: float, sr: int):
    events = events - start_time + offset
    events = events[events < end_time - start_time]
    events *= sr
    return events


def split_signal(signal: torch.Tensor, n_samples_per_frame: int):
    signals = signal.split(n_samples_per_frame)

    if len(signals[-1]) != len(signals[0]):
        signals = signals[:-1]

    signals = np.array([s.numpy() for s in signals])
    return signals


def get_binary_labels(events: np.array, n_labels: int, n_samples_per_frame: int):
    labels = []
    for i in range(n_labels):
        t = i * n_samples_per_frame
        counts = (t <= events) & (events < t + n_samples_per_frame)
        counts = counts.sum()
        labels.append(int(counts > 0))
    return np.array(labels)


def get_counts_labels(events: np.array, n_labels: int, n_samples_per_frame: int):
    labels = []
    for i in range(n_labels):
        t = i * n_samples_per_frame
        counts = (t <= events) & (events < t + n_samples_per_frame)
        counts = counts.sum()
        labels.append(counts)
    return np.array(labels)


def under_sampling(signals, labels, seed=0):
    mask = labels != 0

    pos_labels = labels[mask]
    pos_signals = signals[mask]

    neg_labels = labels[~mask]
    neg_signals = signals[~mask]

    np.random.seed(seed)

    neg_random_idx = np.random.choice(range(len(neg_signals)), len(pos_labels))
    neg_labels = neg_labels[neg_random_idx]
    neg_signals = neg_signals[neg_random_idx]

    labels = np.concatenate((pos_labels, neg_labels))
    signals = np.concatenate((pos_signals, neg_signals))

    return signals, labels


def create_simple_dataset(signal, sr, manual_events, from_time, till_time, window_length=10, n_samples=100, seed=42, margin=1, return_timestamps=False):
    np.random.seed(seed)

    samples = []
    labels = []
    timestamps = []
    
    while len(samples) < n_samples:
        sample_from = np.random.rand(1)[0] * (till_time - from_time - window_length) + from_time
        sample_till = sample_from + window_length

        events_timestamps = []
        skip = False
        
        for manual_event in manual_events:
            if sample_from <= manual_event <= sample_till:

                events_timestamps.append(manual_event)
                
                # skip interval with event in margin
                if manual_event < sample_from + margin or manual_event > sample_till - margin:
                    skip = True
                    break        

        if skip:
            continue

        sample = signal[int(sample_from * sr): int(sample_till * sr)]

        samples.append(sample)
        labels.append(len(events_timestamps))
        timestamps.append(events_timestamps)
    
    if return_timestamps:
        return samples, labels, timestamps
    
    return samples, labels

def create_transformation(params):
    melkwargs = {
        "n_fft": params.n_fft,
        "n_mels": params.n_mels,
        "hop_length": params.hop_length
    }

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=params.sr,
        **melkwargs
    )

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=params.sr,
        n_mfcc=params.n_mfcc,
        melkwargs=melkwargs
    )

    transform = lambda signal: torch.cat((mel_transform(signal), mfcc_transform(signal)), dim=0).unsqueeze(0)

    return transform