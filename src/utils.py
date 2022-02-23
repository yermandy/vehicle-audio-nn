import os
import wandb
import math
from tabulate import tabulate
import torch
import torchaudio
import torch.nn as nn
import numpy as np
import omegaconf

from easydict import EasyDict
from tqdm.auto import tqdm
from datetime import datetime
from copy import deepcopy

from .video import Video
from .datapool import DataPool
from .loaders import *
from .constants import *
from .params import *


def get_device(cuda):
    return torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')

def get_cumsum(T, E):
    hist = []
    for i in range(1, len(T)):
        N = np.sum((E >= T[i - 1]) & (E < T[i]))
        hist.append(N)
    cumstep = np.cumsum(hist)
    return cumstep


def get_intervals_from_files(files, from_time, till_time):
    _intervals = []
    _events_in_intervals = []
    for file in files:
        print(f'loading: {file}')
        signal, sr = load_audio(file, return_sr=True)
        intervals = load_intervals(file)
        intervals, events_in_intervals = preprocess_intervals(intervals, from_time, till_time)
        
        for interval in intervals:
            _intervals.append(signal[int(interval[0] * sr): int(interval[1] * sr)])
        
        _events_in_intervals.extend(events_in_intervals)

    return _intervals, np.array(_events_in_intervals)


def preprocess_intervals(intervals, from_time, till_time):
    intervals, events_in_intervals = intervals[0], intervals[1]
    if from_time is not None and till_time is not None:
        mask = (intervals >= from_time) & (intervals < till_time)
        intervals = intervals[mask]
        events_in_intervals = events_in_intervals[mask]
    
    _intervals = []
    _events_in_intervals = []
    for i in range(1, len(intervals)):
        _intervals.append([intervals[i - 1], intervals[i]])
        _events_in_intervals.append(events_in_intervals[i])

    return _intervals, np.array(_events_in_intervals)


def crop_events(events, from_time, till_time):
    events = deepcopy(events)
    return events[(events >= from_time) & (events < till_time)]


def crop_signal(signal, sr, from_time, till_time):
    signal = deepcopy(signal)
    return signal[int(from_time * sr): int(till_time * sr)]


def crop_signal_events(signal, events, sr, from_time, till_time):
    signal = deepcopy(signal)
    events = deepcopy(events)

    if from_time is not None and till_time is not None:
        signal = crop_signal(signal, sr, from_time, till_time)
        
        if events is not None:
            events = crop_events(events, from_time, till_time)

    if events is None:
        return signal

    return signal, events


def get_time(signal, params, from_time, till_time):
    n_hops = get_n_hops(signal, params)
    time = np.linspace(from_time, till_time, n_hops + 1)
    return time


def get_diff(signal, events, predictions, params, from_time=None, till_time=None):
    if from_time == None:
        from_time = 0
    
    if till_time == None:
        till_time = get_signal_length(signal, params)

    signal, events = crop_signal_events(signal, events, params.sr, from_time, till_time)

    time = get_time(signal, params, from_time, till_time)

    cumsum_pred = np.cumsum(predictions)
    cumsum_true = get_cumsum(time, events)
    return np.abs(cumsum_pred - cumsum_true).mean()


def get_n_hops(signal, params):
    n_samples = len(signal)

    if 'n_samples_in_window' not in params or 'n_samples_in_nn_hop' not in params:
        params = get_additional_params(params)

    # TODO double check this
    n_hops = n_samples // params.n_samples_in_nn_hop

    return n_hops


def get_labels(events, window_length, from_time, till_time):
    events = crop_events(events, from_time, till_time)
    hops = int((till_time - from_time) // window_length)
    labels = []
    for i in range(1, hops + 1):
        mask = events < window_length * i
        events = events[~mask]
        labels.append(mask.sum())
    labels = np.array(labels)
    return labels


def get_signal_length(signal, config):
    # in this project we assume that the loaded signal has 44100 samples in one second
    return len(signal) // config.sr


def create_dataset_from_files(datapool: DataPool, window_length=6, n_samples=5000, seed=42, is_trn=True, offset=0):
    """ if n_samples == -1, dataset is created sequentially from a sequence """

    all_samples = []
    all_labels = []
    all_domains = []

    n_files = len(datapool)
    n_samples_per_file = n_samples // n_files

    for i, video in enumerate(datapool):
        video: Video = video
        signal = video.signal
        sr = video.sr
        events = video.events

        from_time, till_time = video.get_from_till_time(is_trn)
        from_time = from_time + offset
        
        if n_samples == -1:
            samples, labels = create_dataset_sequentially(signal, sr, events,
                from_time=from_time, till_time=till_time, window_length=window_length)
        else:
            samples, labels = create_dataset_uniformly(signal, sr, events,
                from_time=from_time, till_time=till_time, seed=seed,
                window_length=window_length, n_samples=n_samples_per_file)


        # print(f'sampled {len(samples)} from {video.file}')
        all_samples.extend(samples)
        all_labels.extend(labels)
        all_domains.extend([i] * len(labels))

    return all_samples, all_labels, all_domains


def create_dataset_uniformly(signal, sr, events, from_time=None, till_time=None, window_length=10, n_samples=100, seed=42, margin=0, return_timestamps=False):

    if from_time is None:
        from_time = 0

    max_time = len(signal) // sr

    if till_time is None or till_time > max_time:
        till_time = max_time

    np.random.seed(seed)

    samples = []
    labels = []
    timestamps = []

    # _all_from = []
    # _all_till = []

    while len(samples) < n_samples:
        sample_from = np.random.rand(1)[0] * (till_time - from_time - window_length) + from_time
        sample_till = sample_from + window_length

        events_timestamps = []
        skip = False

        for event in events:
            if sample_from <= event < sample_till:

                events_timestamps.append(event)

                # skip interval with event in margin
                if event < sample_from + margin or event > sample_till - margin:
                    skip = True
                    break

        if skip:
            continue

        # _all_from.append(sample_from)
        # _all_till.append(sample_till)

        sample = signal[int(sample_from * sr): int(sample_till * sr)]

        samples.append(sample)
        labels.append(len(events_timestamps))
        timestamps.append(events_timestamps)

    # print(np.min(_all_from), np.max(_all_from))
    # print(np.min(_all_till), np.max(_all_till))

    if return_timestamps:
        return samples, labels, timestamps

    return samples, labels


def create_dataset_sequentially(signal, sr, events, from_time=None, till_time=None, window_length=10):

    if from_time is None:
        from_time = 0

    max_time = len(signal) // sr

    if till_time is None or till_time > max_time:
        till_time = max_time

    samples = []
    labels = []

    interval_time = till_time - from_time
    n_samples = int(interval_time // window_length)

    for i in range(n_samples):
        sample_from = from_time + i * window_length
        sample_till = sample_from + window_length

        mask = (events >= sample_from) & (events < sample_till)
        label = mask.sum()

        sample = signal[int(sample_from * sr): int(sample_till * sr)]

        samples.append(sample)
        labels.append(label)

    return samples, labels


def create_fancy_table(outputs):
    rvce = outputs[:, 0].astype(float)
    error = outputs[:, 1].astype(int)
    n_events = outputs[:, 2].astype(int)
    mae = outputs[:, 3].astype(float)

    header = ['rvce', 'error', 'n_events', 'mae', 'time', 'file']
    footer = [
        f'{rvce.mean():.2f} ± {rvce.std():.2f}',
        f'{error.mean():.2f} ± {error.std():.2f}',
        f'{n_events.mean():.2f} ± {n_events.std():.2f}',
        f'{mae.mean():.2f} ± {mae.std():.2f}',
        '',
        'summary'
    ]
    
    table = np.vstack(([header], outputs, [footer]))
    fancy_table = tabulate(table, headers='firstrow', tablefmt='fancy_grid', showindex=True)

    return table, fancy_table

def print_config(config):
    table = [] 
    for k, v in config.items():
        if type(v) == omegaconf.listconfig.ListConfig: 
            table.append([k, f'list with {len(v)} entries'])
        elif type(v) == omegaconf.dictconfig.DictConfig: 
            table.append([k, f'dict with {len(v)} entries'])
        else:
            table.append([k, v])
    print(tabulate(table))

