import os
import wandb
import math
from tabulate import tabulate
import torch
import torchaudio
import torch.nn as nn
import numpy as np
import omegaconf
import csv

from easydict import EasyDict
from tqdm.auto import tqdm
from datetime import datetime
from copy import deepcopy

from .video import Video
from .datapool import DataPool
from .loaders import *
from .constants import *
from .config import *


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
    I = []
    E = []
    for file in files:
        print(f'loading: {file}')
        signal, sr = load_audio(file, return_sr=True)
        intervals = load_intervals(file)
        intervals, events_in_intervals = preprocess_intervals(intervals, from_time, till_time)
        
        for interval in intervals:
            I.append(signal[int(interval[0] * sr): int(interval[1] * sr)])
        
        E.extend(events_in_intervals)

    return np.array(I), np.array(E)


def preprocess_intervals(intervals, from_time, till_time):
    intervals, events_in_intervals = intervals[0], intervals[1]
    if from_time is not None and till_time is not None:
        mask = (intervals >= from_time) & (intervals < till_time)
        intervals = intervals[mask]
        events_in_intervals = events_in_intervals[mask]
    
    I = []
    E = []
    for i in range(1, len(intervals)):
        I.append([intervals[i - 1], intervals[i]])
        E.append(events_in_intervals[i])

    return np.array(I), np.array(E)


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


def get_time(config: Config, from_time: float, till_time: float):
    n_hops = get_n_hops(config, from_time, till_time)
    time = np.linspace(from_time, till_time, n_hops + 1)
    return time


def get_diff(signal: torch.Tensor, events: np.ndarray, predictions: np.ndarray, config: Config, from_time: float = None, till_time: float = None):
    if from_time == None:
        from_time = 0
    
    if till_time == None:
        till_time = get_signal_length(signal, config)

    signal, events = crop_signal_events(signal, events, config.sr, from_time, till_time)

    time = get_time(config, from_time, till_time)

    cumsum_pred = np.cumsum(predictions)
    cumsum_true = get_cumsum(time, events)
    return np.abs(cumsum_pred - cumsum_true).mean()


def get_n_hops(config: Config, from_time, till_time) -> int:
    return int((till_time - from_time) // config.nn_hop_length)


def get_directions_dict() -> dict:
    return {
        'n_incoming': 'frontal',
        'n_outgoing': 'rear',
    }

def get_categories_dict() -> dict:
    return {
        'n_BUS': 'BUS',
        'n_CAR': 'CAR',
        'n_ERR': 'ERR',
        'n_HVT': 'HVT',
        'n_LGT': 'LGT',
        'n_MTB': 'MTB',
        'n_UNK': 'UNK',
        'n_UNL': 'UNL',
        'n_VAN': 'VAN',
        'n_TO12': 'TO12',
        'n_CYCLE': 'CYCLE',
        'n_TRUCK': 'TRUCK',
        'n_TO34': 'TO34',
        'n_SPECIAL': 'SPECIAL',
        'n_PEDESTRIAN': 'PEDESTRIAN',
        'n_TO12_CARAVAN': 'TO12_CARAVAN'
    }


def _extract_labels(video: Video, labels, mask):
    n_counts = mask.sum()
    labels['n_counts'].append(n_counts)
    labels['domain'].append(video.domain)

    for head_name, categoty_name in get_directions_dict().items():
        if head_name in video.config.heads:
            label = np.sum(video.views[mask] == categoty_name)
            labels[head_name].append(label)

    for head_name, categoty_name in get_categories_dict().items():
        if head_name in video.config.heads:
            label = np.sum(video.category[mask] == categoty_name)
            labels[head_name].append(label)


def get_labels(video: Video, from_time, till_time) -> np.ndarray:
    n_hops = get_n_hops(video.config, from_time, till_time)
    labels = defaultdict(lambda: [])

    for i in range(n_hops):
        sample_from = from_time + i * video.config.nn_hop_length
        sample_till = sample_from + video.config.window_length
        mask = (video.events >= sample_from) & (video.events < sample_till)
        _extract_labels(video, labels, mask)
    
    labels = {k: np.array(v) for k, v in labels.items()}
    return labels


def get_signal_length(signal, config):
    return len(signal) // config.sr


def create_dataset_from_files(datapool: DataPool, part=Part.LEFT, offset: float=0):
    all_samples = []
    all_labels = defaultdict(lambda: [])

    for i, video in enumerate(datapool):
        video: Video = video
        video.domain = i

        from_time, till_time = video.get_from_till_time(part)
        from_time = from_time + offset
        
        samples, labels = create_dataset_sequentially(video, from_time=from_time, till_time=till_time)

        for k, v in labels.items():
            all_labels[k].extend(v)

        all_samples.extend(samples)

    return all_samples, all_labels


def create_dataset_sequentially(video: Video, from_time=None, till_time=None):
    if from_time is None:
        from_time = 0

    max_time = len(video.signal) // video.sr

    if till_time is None or till_time > max_time:
        till_time = max_time

    samples = []
    labels = defaultdict(lambda: [])

    interval_time = till_time - from_time
    n_samples = int(interval_time // video.config.window_length)

    for i in range(n_samples):
        sample_from = from_time + i * video.config.window_length
        sample_till = sample_from + video.config.window_length
        mask = (video.events >= sample_from) & (video.events < sample_till)
        _extract_labels(video, labels, mask)

        sample = video.signal[int(sample_from * video.sr): int(sample_till * video.sr)]
        samples.append(sample)

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
        if isinstance(v, (list, omegaconf.listconfig.ListConfig)):
            table.append([k, f'list with {len(v)} entries'])
        elif isinstance(v, (dict, omegaconf.dictconfig.DictConfig)):
            table.append([k, f'dict with {len(v)} entries'])
        elif callable(v):
            table.append([k, f'function'])
        else:
            table.append([k, v])
    print(tabulate(table))



def save_dict_csv(name: str, dict: Dict[str, np.ndarray]):
    with open(name, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(dict.keys())
        rows = np.array(list(dict.values())).T
        writer.writerows(rows)


def save_dict_txt(name: str, dict: Dict[str, np.ndarray]):
    with open(name, 'w') as file:
        table = tabulate(dict, headers='keys', tablefmt='fancy_grid', showindex=True)
        file.write(table)