import os
from typing import Callable
import wandb
import math
from tabulate import tabulate
import torch
import torchaudio
import torch.nn as nn
import numpy as np
import omegaconf
import csv
import pandas as pd
import contextlib
import time


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
    return torch.device(
        f"cuda:{cuda}" if torch.cuda.is_available() and cuda >= 0 else "cpu"
    )


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
        print(f"loading: {file}")
        signal, sr = load_audio(file, return_sr=True)
        intervals = load_intervals(file)
        intervals, events_in_intervals = preprocess_intervals(
            intervals, from_time, till_time
        )

        for interval in intervals:
            I.append(signal[int(interval[0] * sr) : int(interval[1] * sr)])

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
    return signal[int(from_time * sr) : int(till_time * sr)]


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


def get_diff(
    signal: torch.Tensor,
    events: np.ndarray,
    predictions: np.ndarray,
    config: Config,
    from_time: float = None,
    till_time: float = None,
):
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
        "n_incoming": "frontal",
        "n_outgoing": "rear",
    }


def get_categories_dict() -> Dict[str, str]:
    return {
        "n_BUS": "BUS",
        "n_CAR": "CAR",
        "n_ERR": "ERR",
        "n_HVT": "HVT",
        "n_LGT": "LGT",
        "n_MTB": "MTB",
        "n_UNK": "UNK",
        "n_UNL": "UNL",
        "n_VAN": "VAN",
        "n_TO12": "TO12",
        "n_CYCLE": "CYCLE",
        "n_TRUCK": "TRUCK",
        "n_TO34": "TO34",
        "n_SPECIAL": "SPECIAL",
        "n_PEDESTRIAN": "PEDESTRIAN",
        "n_TO12_CARAVAN": "TO12_CARAVAN",
    }


def get_categories_functions_dict() -> Dict[str, Callable]:
    return {"n_NOT_CAR": lambda array: array != "CAR"}


def _extract_labels(video: Video, labels, mask):
    n_counts = mask.sum()
    labels["n_counts"].append(n_counts)
    labels["domain"].append(video.domain)

    for head_name, category_name in get_directions_dict().items():
        if head_name in video.config.heads:
            label = np.sum(video.views[mask] == category_name)
            labels[head_name].append(label)

    for head_name, category_name in get_categories_dict().items():
        if head_name in video.config.heads:
            label = np.sum(video.category[mask] == category_name)
            labels[head_name].append(label)

    for head_name, category_function in get_categories_functions_dict().items():
        if head_name in video.config.heads:
            label = np.sum(category_function(video.category[mask]))
            labels[head_name].append(label)


def get_labels(video: Video, from_time, till_time) -> np.ndarray:
    n_hops = get_n_hops(video.config, from_time, till_time)
    labels = defaultdict(lambda: [])

    mod_events = np.mod(video.events, 6)
    center = video.config.window_length / 2

    for i in range(n_hops):
        sample_from = from_time + i * video.config.nn_hop_length
        sample_till = sample_from + video.config.window_length
        mask = (video.events >= sample_from) & (video.events < sample_till)
        _extract_labels(video, labels, mask)

        dist_to_center = np.abs(mod_events[mask] - center)
        if len(dist_to_center) > 0:
            labels["dist_to_center"].append(dist_to_center.max())
        else:
            labels["dist_to_center"].append(-1)

    labels = {k: np.array(v) for k, v in labels.items()}
    return labels


def get_signal_length(signal, sr_or_config):
    if type(sr_or_config) == Config:
        return len(signal) // sr_or_config.sr
    else:
        return len(signal) // sr_or_config


def create_dataset_from_datapool(datapool: DataPool, part=Part.LEFT, offset: float = 0):
    all_samples = []
    all_labels = defaultdict(lambda: [])

    for i, video in enumerate(datapool):
        video: Video = video
        video.domain = i

        from_time, till_time = video.get_from_till_time(part)
        from_time = from_time + offset

        samples, labels = create_dataset_from_video(
            video, from_time=from_time, till_time=till_time
        )

        for k, v in labels.items():
            all_labels[k].extend(v)

        all_samples.extend(samples)

    return all_samples, all_labels


def create_samples(config: Config, signal, from_time, till_time):
    samples = []
    n_hops = get_n_hops(config, from_time, till_time)
    for i in range(n_hops):
        sample_from = from_time + i * config.nn_hop_length
        sample_till = sample_from + config.window_length
        sample = signal[int(sample_from * config.sr) : int(sample_till * config.sr)]
        samples.append(sample)
    return samples


def create_dataset_from_video(video: Video, from_time=None, till_time=None):
    if from_time == None and till_time == None:
        from_time, till_time = video.get_from_till_time(Part.WHOLE)

    if from_time is None:
        from_time = 0

    max_time = len(video.signal) // video.sr

    if till_time is None or till_time > max_time:
        till_time = max_time

    samples = []
    labels = defaultdict(lambda: [])

    n_hops = get_n_hops(video.config, from_time, till_time)

    for i in range(n_hops):
        sample_from = from_time + i * video.config.nn_hop_length
        sample_till = sample_from + video.config.window_length

        mask = (video.events >= sample_from) & (video.events < sample_till)
        _extract_labels(video, labels, mask)

        sample = video.signal[int(sample_from * video.sr) : int(sample_till * video.sr)]
        samples.append(sample)

    return samples, labels


def print_config(config):
    table = []
    for k, v in config.items():
        if isinstance(v, (list, omegaconf.listconfig.ListConfig)):
            table.append([k, f"list with {len(v)} entries"])
        elif isinstance(v, (dict, omegaconf.dictconfig.DictConfig)):
            table.append([k, f"dict with {len(v)} entries"])
        elif callable(v):
            table.append([k, f"function"])
        else:
            table.append([k, v])
    print(tabulate(table))


def create_subfolders(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def save_dict_csv(path: str, dict: Dict[str, np.ndarray]):
    create_subfolders(path)
    pd.DataFrame.from_dict(dict).to_csv(path, index=False)
    # with open(name, "w") as file:
    #     writer = csv.writer(file)
    #     writer.writerow(dict.keys())
    #     rows = np.array(list(dict.values())).T
    #     writer.writerows(rows)


def save_dict_txt(path: str, dict: Dict[str, np.ndarray]):
    create_subfolders(path)
    with open(path, "w") as file:
        table = tabulate(dict, headers="keys", tablefmt="fancy_grid", showindex=True)
        file.write(table)


def generate_cross_validation_table(uuids, model_name="rvce", prefix="tst"):
    root_uuid = uuids[0].split("/")[0]
    table = []
    header = []
    dict = {}
    for uuid in uuids:
        results = np.genfromtxt(
            f"outputs/{uuid}/results/{prefix}_{model_name}_output.csv",
            delimiter=",",
            skip_footer=1,
            dtype=str,
        )
        header = results[0]
        results = results[1:]
        results = np.atleast_2d(results)
        table.extend(results)
    table = np.array(table).T
    times = []
    files = []
    for i in range(len(header)):
        column_name = header[i]
        column = table[i].tolist()
        if column_name == "file":
            files = column
        elif column_name == "time":
            times = column
        else:
            dict[column_name] = column

    append_summary(dict, times, files)
    save_dict_csv(f"outputs/{root_uuid}/{prefix}_{model_name}_output.csv", dict)
    save_dict_txt(f"outputs/{root_uuid}/{prefix}_{model_name}_output.txt", dict)


def append_summary(dict, times, files):
    for k, v in dict.items():
        v = np.array(v).astype(float)
        stats = f"{v.mean():.3f} ± {v.std():.3f}"
        dict[k].append(stats)

    times.append("")
    files.append("summary")

    dict["time"] = times
    dict["file"] = files


def aslist(x):
    return x if isinstance(x, list) else [x]


def m(s: float) -> float:
    return s * 60


def h(s: float) -> float:
    return s * 3600


class Profile(contextlib.ContextDecorator):
    # Usage: @Profile() decorator or 'with Profile():' context manager

    def __init__(self, t=0.0):
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self._dt = self.time() - self.start
        self.t += self._dt

    def time(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()
