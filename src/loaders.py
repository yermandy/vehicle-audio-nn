import torch
import torchaudio
import numpy as np


def time_to_sec(time):
    h, m, s = map(float, time.split(':'))
    sec = h * 3600 + m * 60 + s
    return sec


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