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


def load_audio(audio_file):
    signal, _ = torchaudio.load(audio_file)
    signal = signal.mean(0)
    return signal


def load_events(events_file):
    return np.loadtxt(events_file)


def load_direction_from_csv(csv):
    directions = {}
    for row in csv[1:]:
        plate, direction = row[[0, 7]]
        directions[plate] = direction
    return np.array(list(directions.values()))


def load_events_from_csv(csv):
    events = []
    for e in np.unique(csv[1:, 14]):
        h, m, s = map(float, e.split(':'))
        sec = h * 3600 + m * 60 + s
        events.append(sec)
    return np.array(events)


def load_csv(csv_file):
    import glob
    for f in glob.glob('data/csv/*.csv'):
        name = csv_file.split('/')[-1]
        if f.find(name) != -1:
            csv_file = f
            break
    return np.genfromtxt(csv_file, dtype=str, delimiter=';')


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


def get_diff(results, params):
    cumsum = np.cumsum(results)
    cumstep = get_cumstep(params.time, params.events)
    return np.abs(cumsum - cumstep).mean()


def get_additional_params(params, signal=None, events=None, start_time=0, end_time=0):
    additional = EasyDict()

    if signal is not None:
        # crop signal to interval
        additional.signal = signal[start_time * params.sr: end_time * params.sr]

    if events is not None:
        # take events that are in interval
        additional.events = events[(events >= start_time) & (events < end_time)]

    interval = end_time - start_time

    n_samples_in_nn_hop = int(params.sr * params.nn_hop_length)
    n_samples_in_frame = int(params.sr * params.frame_length)
    n_samples_in_interval = int(params.sr * interval)

    n_features_in_sec = params.sr // params.hop_length
    n_features_in_nn_hop = int(n_features_in_sec * params.nn_hop_length)
    n_features_in_frame = int(n_features_in_sec * params.frame_length)
    n_features_in_interval = int(n_features_in_sec * interval)

    # number of hops
    n_hops = (n_samples_in_interval - n_samples_in_frame) // n_samples_in_nn_hop
        
    # create time axis for prediction visualization
    nn_hop_length_half = params.nn_hop_length // 2
    time = np.linspace(start_time + nn_hop_length_half, end_time - nn_hop_length_half, n_hops)

    additional.n_hops = n_hops
    additional.time = time
    additional.n_samples_in_nn_hop = n_samples_in_nn_hop
    additional.n_samples_in_frame = n_samples_in_frame
    additional.n_features_in_nn_hop = n_features_in_nn_hop
    additional.n_features_in_frame = n_features_in_frame
    additional.n_features_in_interval = n_features_in_interval

    return additional



def validate_2(model, loader, tqdm=lambda x: x):
    device = next(model.parameters()).device

    results = []

    loop = tqdm(loader)

    model.eval()
    with torch.no_grad():
        for x in loop:
            x = x.to(device)
            y = model(x).squeeze().tolist()
            results.extend(y)

    results = np.array(results)
    return results


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
                y = model(batch).squeeze().tolist()
                results.extend(y)
                batch = []

    results = np.array(results)
    return results


def validate_multi(model, dataset, params, tqdm=lambda x: x):
    device = next(model.parameters()).device
    results = []

    loop = tqdm(range(params.n_hops))

    model.eval()
    with torch.no_grad():
        for k in loop:
            start = k * params.n_samples_in_nn_hop
            end = start + params.n_samples_in_frame
            x = params.signal[start:end]
            x = dataset.transform(x).unsqueeze(0)
            x = x.to(device)
            y = model(x).argmax(1).item()
            results.append(y)

    results = np.array(results)
    return results