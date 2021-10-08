import os
import wandb
import math
import torch
import torchaudio
import torch.nn as nn
import numpy as np

from easydict import EasyDict
from tqdm import tqdm
from datetime import datetime


def load_csv(name, folder='data/csv/*.csv'):
    import glob
    batch = []
    for file in glob.glob(folder):
        if file.find(name) != -1:
            table = np.genfromtxt(file, dtype=str, delimiter=';', skip_header=1)
            batch.append(table)
    batch = np.concatenate(batch)
    return batch


def load_audio(audio_file, return_sr=False):
    signal, sr = torchaudio.load(audio_file)
    signal = signal.mean(0)
    # round to the last second
    seconds = len(signal) // sr
    signal = signal[:seconds * sr]
    if return_sr:
        return signal, sr
    return signal


def time_to_sec(time):
    h, m, s = map(float, time.split(':'))
    sec = h * 3600 + m * 60 + s
    return sec


def load_events(file):
    return np.loadtxt(file)


def load_intervals(file):
    arr = np.loadtxt(file)
    intervals, events_in_interval = arr[:, 0], arr[:, 1]
    return intervals, events_in_interval


def load_column(csv, column):
    out = {}
    for row in csv:
        out[row[0]] = row[column]
    return np.array(list(out.values()))


def load_directions_from_csv(csv):
    return load_column(csv, 7)


def load_views_from_csv(csv):
    return load_column(csv, 23)


def load_events_from_csv(csv):
    return np.array([time_to_sec(t) for t in load_column(csv, 14)])


def load_event_start_time_from_csv(csv):
    start_times = {}
    for row in csv:
        detection_id, time = row[[0, 8]]
        start_times[detection_id] = time
    start_times = {k: time_to_sec(v) for k, v in start_times.items()}
    return np.array(list(start_times.values()))


def load_event_time_from_csv(csv):
    times = {}
    for row in csv:
        detection_id, start_time, end_time = row[[0, 8, 9]]
        times[detection_id] = start_time, end_time
    
    start_times = []
    end_times = []
    for k, v in times.items():
        start_time, end_time = v
        start_time = time_to_sec(start_time)
        try:
            end_time = time_to_sec(end_time)
        except:
            end_time = start_time

        start_times.append(start_time)
        end_times.append(end_time)
    return np.array(start_times), np.array(end_times)


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


def crop_signal_events(signal, events, sr, from_time, till_time):
    if from_time is not None and till_time is not None:
        signal = signal[from_time * sr: till_time * sr]
        
        if events is not None:
            events = events[(events >= from_time) & (events < till_time)]

    if events is None:
        return signal

    return signal, events


def get_time(signal, params, from_time, till_time):
    nn_hop_length_half = params.nn_hop_length // 2
    n_hops = get_n_hops(signal, params)
    time = np.linspace(from_time + nn_hop_length_half, till_time - nn_hop_length_half, n_hops)
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

    # n_hops = (n_samples - params.n_samples_in_frame) // params.n_samples_in_nn_hop
    n_hops = int(np.ceil((n_samples - params.n_samples_in_frame) / params.n_samples_in_nn_hop))

    return n_hops
    

def get_additional_params(params):

    # if signal is not None:
        # crop signal to interval
        # params.signal = signal[start_time * params.sr: end_time * params.sr]

    # if events is not None:
        # take events that are in interval
        # params.events = events[(events >= start_time) & (events < end_time)]

    # interval = end_time - start_time

    n_samples_in_nn_hop = int(params.sr * params.nn_hop_length)
    n_samples_in_frame = int(params.sr * params.frame_length)
    # n_samples_in_interval = int(params.sr * interval)

    n_features_in_sec = params.sr // params.hop_length
    n_features_in_nn_hop = int(n_features_in_sec * params.nn_hop_length)
    n_features_in_frame = int(n_features_in_sec * params.frame_length)
    # n_features_in_interval = int(n_features_in_sec * interval)

    # number of hops
    # n_hops = (n_samples_in_interval - n_samples_in_frame) // n_samples_in_nn_hop
        
    # create time axis for prediction visualization
    # nn_hop_length_half = params.nn_hop_length // 2
    # time = np.linspace(start_time + nn_hop_length_half, end_time - nn_hop_length_half, n_hops)

    # params.n_hops = n_hops
    # params.time = time
    params.n_samples_in_nn_hop = n_samples_in_nn_hop
    params.n_samples_in_frame = n_samples_in_frame
    params.n_features_in_nn_hop = n_features_in_nn_hop
    params.n_features_in_frame = n_features_in_frame
    # params.n_features_in_interval = n_features_in_interval

    return params


def validate(model, dataset, params, tqdm=lambda x: x, batch_size=32):
    device = next(model.parameters()).device

    batch = []
    results = []

    loop = tqdm(range(params.n_hops))

    model.eval()
    with torch.no_grad():
        for k in loop:
            start = k * params.n_samples_in_nn_hop
            end = start + params.n_samples_in_frame
            x = params.signal[start:end]
            x = dataset.transform(x)
            batch.append(x)
            
            if (k + 1) % batch_size == 0 or k + 1 == params.n_hops:
                batch = torch.stack(batch, dim=0)
                batch = batch.to(device)
                y = model(batch).view(-1).tolist()
                results.extend(y)
                batch = []

    results = np.array(results)
    return results


def validate_multi(signal, model, transform, params, tqdm=lambda x: x, batch_size=32, return_probs=False, from_time=None, till_time=None):

    signal = crop_signal_events(signal, None, params.sr, from_time, till_time)
        
    device = next(model.parameters()).device

    batch = []
    results = []
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
                # print((model(batch).softmax(1) * 100).round())
                scores = model(batch)
                
                if return_probs:
                    p = scores.softmax(1).tolist()
                    probs.extend(p)

                y = scores.argmax(1).view(-1).tolist()
                results.extend(y)
                batch = []

    results = np.array(results)

    if return_probs:
        probs = np.array(probs)
        return results, probs

    return results


def validate_intervals(intervals, events_in_intervals, model, transform, params):
    error = 0
    for signal, n_events in zip(intervals, events_in_intervals):
        results = validate_multi(signal, model, transform, params)
        pred = np.cumsum(results)[-1]
        error += np.abs(pred - n_events)
    return error / len(intervals)