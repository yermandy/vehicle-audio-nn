import torch
import numpy as np
import random


def get_offset(max_offset: float):
    return random.uniform(0, max_offset)


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
