import os
from torch.nn.modules.module import T
import wandb
import math
import torch
import torchaudio
import torch.nn as nn
import numpy as np

from easydict import EasyDict
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy

from .video import Video
from .datapool import DataPool
from .loaders import *


def get_split_indices(params):
    n_features_in_sec = params.sr / params.hop_length
    n_features_in_nn_hop = math.ceil(params.nn_hop_length * n_features_in_sec)
    n_features_in_frame = math.ceil(params.frame_length * n_features_in_sec)

    split_indices = []
    for f in range(params.n_frames):
        start = f * n_features_in_nn_hop
        end = f * n_features_in_nn_hop + n_features_in_frame
        split_indices.append([start, end])

    return split_indices


def get_cumstep(T, E):
    closest = lambda array, value: np.abs(array - value).argmin()

    cumstep = np.zeros_like(T)
    for e in E:
        idx = closest(T, e)
        cumstep[idx] += 1
    return np.cumsum(cumstep)


def get_intervals_from_files(files, from_time, till_time):
    _intervals = []
    _events_in_intervals = []
    for file in files:
        print(f'loading: {file}')
        intervals_file = f'data/intervals/{file}.MP4.txt'
        signal, sr = load_audio(f'data/audio/{file}.MP4.wav', return_sr=True)
        intervals = load_intervals(intervals_file)
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
    nn_hop_length_half = params.nn_hop_length // 2
    n_hops = get_n_hops(signal, params)
    # TODO check this
    # time = np.linspace(from_time + nn_hop_length_half, till_time - nn_hop_length_half, n_hops)
    time = np.linspace(from_time, till_time, n_hops)
    return time


def get_diff(signal, events, results, params, from_time=None, till_time=None):
    if from_time == None:
        from_time = 0
    
    if till_time == None:
        till_time = len(signal) / params.sr

    signal, events = crop_signal_events(signal, events, params.sr, from_time, till_time)

    time = get_time(signal, params, from_time, till_time)

    cumsum = np.cumsum(results)
    cumstep = get_cumstep(time, events)
    return np.abs(cumsum - cumstep).mean()


def get_n_hops(signal, params):
    
    if 'n_samples_in_frame' not in params or 'n_samples_in_nn_hop' not in params:
        params = get_additional_params(params)

    n_samples = len(signal)

    # TODO double check this
    n_hops = n_samples // params.n_samples_in_nn_hop

    return n_hops
    

def get_additional_params(params):
    n_samples_in_nn_hop = int(params.sr * params.nn_hop_length)
    n_samples_in_frame = int(params.sr * params.frame_length)
    n_features_in_sec = params.sr // params.hop_length
    n_features_in_nn_hop = int(n_features_in_sec * params.nn_hop_length)
    n_features_in_frame = int(n_features_in_sec * params.frame_length)
    params.n_samples_in_nn_hop = n_samples_in_nn_hop
    params.n_samples_in_frame = n_samples_in_frame
    params.n_features_in_nn_hop = n_features_in_nn_hop
    params.n_features_in_frame = n_features_in_frame
    return params


def validate(signal, model, transform, params, tqdm=lambda x: x, batch_size=32, return_probs=False, from_time=None, till_time=None, classification=True):

    if from_time is not None and till_time is not None:
        signal = crop_signal(signal, params.sr, from_time, till_time)
        
    device = next(model.parameters()).device

    batch = []
    predictions = []
    probs = []

    n_hops = get_n_hops(signal, params) 

    loop = tqdm(range(n_hops))

    model.eval()
    with torch.no_grad():
        for k in loop:
            start = k * params.n_samples_in_nn_hop
            end = start + params.n_samples_in_frame
            x = signal[start: end]
            x = transform(x)
            batch.append(x)
            
            if (k + 1) % batch_size == 0 or k + 1 == n_hops:
                batch = torch.stack(batch, dim=0)
                batch = batch.to(device)
                scores = model(batch)
                
                if return_probs:
                    p = scores.softmax(1).tolist()
                    probs.extend(p)

                if classification:
                    y = scores.argmax(1).view(-1).tolist()
                else:
                    y = scores.view(-1).tolist()

                predictions.extend(y)
                batch = []

    predictions = np.array(predictions)

    if return_probs:
        probs = np.array(probs)
        return predictions, probs

    return predictions


def validate_intervals(datapool: DataPool, is_trn: bool, model, transform, params, classification=True):
    interval_error = 0
    difference_error = 0

    n_intervals = 0
    for video in datapool:
        video: Video = video
        n_events = video.get_events_count(is_trn)
        from_time, till_time = video.get_from_till_time(is_trn)

        predictions = validate(video.signal, model, transform, params, from_time=from_time, till_time=till_time, classification=classification)
        n_intervals += 1

        # calculate error at the end of interval
        interval_error += np.abs(predictions.sum() - n_events) / n_events

        # calculate cumulative histogram difference
        difference_error += get_diff(video.signal, video.events, predictions, params, from_time, till_time)

    mean_interval_error = interval_error / n_intervals
    mean_difference_error = difference_error / n_intervals

    return mean_interval_error, mean_difference_error


def get_window_length(params):
    return params.nn_hop_length * (params.n_frames - 1) + params.frame_length


def create_dataset_from_files(datapool: DataPool, window_length=6, n_samples=5000, seed=42, is_trn=True):
    """ if n_samples == -1, dataset is created sequentially from a sequence """

    all_samples = []
    all_labels = []

    n_files = len(datapool)
    n_samples_per_file = n_samples // n_files

    for video in datapool:
        video: Video = video
        signal = video.signal
        sr = video.sr
        events = video.events

        from_time, till_time = video.get_from_till_time(is_trn)
        
        if n_samples == -1:
            samples, labels = create_dataset_sequentially(signal, sr, events,
                from_time=from_time, till_time=till_time, window_length=window_length)
        else:
            samples, labels = create_dataset_uniformly(signal, sr, events,
                from_time=from_time, till_time=till_time, seed=seed,
                window_length=window_length, n_samples=n_samples_per_file)


        print(f'sampled {len(samples)} from {video.file}')
        all_samples.extend(samples)
        all_labels.extend(labels)

    return all_samples, all_labels


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

        events_timestamps = []

        for event in events:
            if sample_from <= event < sample_till:
                events_timestamps.append(event)

        sample = signal[int(sample_from * sr): int(sample_till * sr)]

        samples.append(sample)
        labels.append(len(events_timestamps))

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

    def transform(signal): return torch.cat(
        (mel_transform(signal), mfcc_transform(signal)), dim=0).unsqueeze(0)

    return transform
