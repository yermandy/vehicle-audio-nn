import torch
import torchaudio
import numpy as np
from collections import defaultdict, Counter
import pickle
import os
import yaml
import torch.nn as nn

from .rawnet import RawNet2Architecture
from .model import ResNet18, ResNet34, ResNet50, ResNet1D, Transformer, WaveCNN
import torchaudio.transforms as T
from .constants import *
from .config import *

from typing import Tuple, Any
from glob import glob


def find_wav(file, raise_exception=False):
    return find_path(f'data/audio_wav/**/{file}.wav', raise_exception)


def find_pt(file, raise_exception=False):
    return find_path(f'data/audio_pt/**/{file}.pt', raise_exception)


def find_csv(file, raise_exception=False):
    return find_path(f'data/csv/**/{file}.csv', raise_exception)
    

def find_labels(file, raise_exception=False):
    return find_path(f'data/labels/**/{file}.txt', raise_exception)


def find_manual_counts(file, raise_exception=False):
    return find_path(f'data/manual_counts/**/{file}.txt', raise_exception)


def find_intervals(file, raise_exception=False):
    return find_path(f'data/intervals/**/{file}.txt', raise_exception)


def find_video(file, raise_exception=False):
    return find_path(f'data/video/**/{file}.*', raise_exception)


def time_to_sec(time):
    h, m, s = map(float, time.split(':'))
    sec = h * 3600 + m * 60 + s
    return sec


def get_file_name(path):
    return os.path.basename(path).split('.')[0]


def get_file_extension(path):
    return os.path.basename(path).split('.')[-1]


def find_path(query, raise_exception=False):
    results = glob(query, recursive=True)
    if len(results) == 0:
        if raise_exception:
            raise Exception(f'file "{query}" does not exist')
        else:
            return None
    elif len(results) == 1:
        return results[0]
    elif raise_exception:
        raise Exception(f'found multiple results for "{query}"')
    else:
        print(f'found multiple results for "{query}"')
        print(results)
        return results[0]



def load_csv(file, preprocess=True):
    file_path = find_csv(file, True)
    csv = np.genfromtxt(file_path, dtype=str, delimiter=';', skip_header=1)
    csv = np.atleast_2d(csv)
    if csv.size == 0:
        return []
    if preprocess:
        return preprocess_csv(csv)
    return csv


def load_audio_wav(path, return_sr=False):
    signal, sr = torchaudio.load(path)
    assert sr == 44100, 'sampling rate of the device is not 44100'
    signal = signal.mean(0)
    if return_sr:
        return signal, sr
    return signal


def load_audio_tensor(path, return_sr=False):
    signal, sr = torch.load(path)
    assert sr == 44100, 'sampling rate of the device is not 44100'
    if return_sr:
        return signal, sr
    return signal


def load_audio(file, resample_sr=44100, return_sr=False, normalize=False) -> torch.Tensor:
    wav_file_path = find_wav(file)
    pt_file_path = find_pt(file)
    if pt_file_path:
        signal, sr = load_audio_tensor(pt_file_path, True)
    elif wav_file_path:
        signal, sr = load_audio_wav(wav_file_path, True)
    else:
        raise Exception(f'file "{file}" does not exist')
    if sr != resample_sr:
        signal = T.Resample(sr, resample_sr).forward(signal)
    # round to the last second
    seconds = len(signal) // resample_sr
    signal = signal[:seconds * resample_sr]
    if normalize:
        signal = (signal - signal.mean()) / signal.std()
    if return_sr:
        return signal, resample_sr
    return signal


def load_manual_counts(file) -> int:
    file_path = find_manual_counts(file)
    if file_path != None:
        return int(np.loadtxt(file_path))
    else:
        return None


def load_events(file):
    return np.loadtxt(find_labels(file))


def load_intervals(file):
    file_path = find_intervals(file)
    if file_path:
        return np.atleast_2d(np.loadtxt(file_path))
    else:
        return []


def find_clusters(X, delta=1*60):
    # X is a sorted array
    X = np.array(X)
    clusters = defaultdict(set)

    for x in X:
        mask = np.abs(X - x) < delta
        if mask[0]:
            cluster_id = 0
        else:
            cluster_id = mask.searchsorted(True)
        for i, m in enumerate(mask):
            if m:
                clusters[cluster_id].add(i)

    clusters = {k: list(v) for k, v in clusters.items()}
    return clusters


def preprocess_csv(csv):
    licence_plates = defaultdict(list)
    
    for row in csv:
        plate_id = row[1]
        start_time = row[CsvColumnID.START_TIME]
        end_time = row[CsvColumnID.END_TIME]
        if start_time != '':
            licence_plates[plate_id].append(
                [time_to_sec(start_time), row]
            )
        if end_time != '':
            licence_plates[plate_id].append(
                [time_to_sec(end_time), row]
            )
    
    tracking_uuid = 0
    rows = []
    for key, values in licence_plates.items():
        car_times = np.array([t[0] for t in values])
        car_rows = np.array([t[1] for t in values])

        indices = np.argsort(car_times)
        car_times = car_times[indices]
        car_rows = car_rows[indices]
        
        clusters = find_clusters(car_times)
    
        for cluster_id, cluster_objects in clusters.items():
            traking_rows = np.array(car_rows[cluster_objects])
            
            most_common_view = Counter(item for item in traking_rows[:, CsvColumnID.VIEWS]).most_common(1)[0][0]
            most_common_color = Counter(item for item in traking_rows[:, CsvColumnID.COLOR]).most_common(1)[0][0]
            most_common_category = Counter(item for item in traking_rows[:, CsvColumnID.CATEGORY]).most_common(1)[0][0]
            
            modified_row = np.full_like(traking_rows[0], '')
            
            start_times = traking_rows[:, CsvColumnID.START_TIME]
            end_times = traking_rows[:, CsvColumnID.END_TIME]
            best_detection_frame_times = traking_rows[:, CsvColumnID.BEST_DETECTION_FRAME_TIME]
            
            times = np.concatenate([start_times, end_times])
            times = np.sort(times)
            times = [t for t in times if t != '']
            
            modified_row[0] = tracking_uuid
            modified_row[1] = key
            tracking_uuid += 1
            modified_row[CsvColumnID.BEST_DETECTION_FRAME_TIME] = best_detection_frame_times[-1]
            modified_row[CsvColumnID.START_TIME] = times[0]
            modified_row[CsvColumnID.END_TIME] = times[-1]
            modified_row[CsvColumnID.VIEWS] = most_common_view
            modified_row[CsvColumnID.CATEGORY] = most_common_category
            modified_row[CsvColumnID.COLOR] = most_common_color
            rows.append(modified_row)
    rows = np.array(rows)
    indices = np.argsort(rows[:, CsvColumnID.START_TIME])
    rows = rows[indices]
    return rows


def load_intervals_and_n_events(file):
    events = load_events(file)
    intervals = load_intervals(file)

    n_events_array = []
    for interval_from_time, interval_till_time in intervals:
        n_events = np.sum((events >= interval_from_time) & (events < interval_till_time))
        n_events_array.append(n_events)
    
    combined = np.hstack([intervals, n_events_array])
    return combined


def load_column(csv, column):
    out = {}
    for row in csv:
        out[row[0]] = row[column]
    return np.array(list(out.values()))


def load_views_from_csv(csv):
    return load_column(csv, CsvColumnID.VIEWS)


def load_category_from_csv(csv):
    return load_column(csv, CsvColumnID.CATEGORY)


def load_best_detection_frame_time_from_csv(csv):
    return np.array([time_to_sec(t) for t in load_column(csv, 14)])


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


# TODO Fix
def load_model_wandb(uuid, wandb_entity, wandb_project, model_name='mae', device=None, classification=True):
    import wandb

    if device is None:
        device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

    api = wandb.Api()
    runs = api.runs(f'{wandb_entity}/{wandb_project}', per_page=5000, order='config.uuid')

    for run in runs: 
        if run.name == str(uuid):
            config = EasyDict(run.config)
            break
    
    weights = torch.load(f'outputs/{uuid}/weights/{model_name}.pth', device)
    model = get_model(config).to(device)
    model.load_state_dict(weights)
    model.eval()
    
    return model, config


def load_config_wandb(uuid, wandb_entity, wandb_project) -> Config:
    import wandb
    
    api = wandb.Api()
    runs = api.runs(f'{wandb_entity}/{wandb_project}', per_page=5000, order='config.uuid')

    for run in runs: 
        if run.name == str(uuid):
            config = Config()
            for k, v in run.config.items():
                config.k = v
            return config


def load_run_wandb(uuid, wandb_entity, wandb_project) -> Config:
    import wandb
    
    api = wandb.Api()
    runs = api.runs(f'{wandb_entity}/{wandb_project}', per_page=5000, order='config.uuid')

    for run in runs: 
        if run.name == str(uuid):
            return run


def load_config_locally(uuid) -> Config:
    with open(f'outputs/{uuid}/config.pickle', 'rb') as f:
        return pickle.load(f)


def get_model(config):
    return {
        'WaveCNN': WaveCNN,
        'ResNet18': ResNet18,
        'ResNet34': ResNet34,
        'ResNet50': ResNet50,
        'ResNet1D': ResNet1D,
        'RawNet2': RawNet2Architecture,
        'Transformer': Transformer
    }[config.architecture](config)


def get_optimizer(model, config):
    return {
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW
    }[config.optimizer](model.parameters(), lr=config.lr)


def get_loss(config, trn_dataset, device):
    if config.loss == 'ClassBalancedCrossEntropy':
        if len(config.heads) > 1:
            raise Exception('Class-Balanced Cross Entropy is not supported for multi-head training.')
        else:
            # https://arxiv.org/pdf/1901.05555.pdf
            classes, samples = np.unique(trn_dataset.labels['n_counts'], return_counts=True)
            samples_per_class = np.ones(config.num_classes)
            samples_per_class[classes] = samples
            effective_num = 1.0 - np.power(config.loss_cbce_beta, samples_per_class)
            weights = (1.0 - config.loss_cbce_beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * len(samples_per_class)
            weights = torch.from_numpy(weights).float().to(device)
            print('ClassBalancedCrossEntropy')
            loss = nn.CrossEntropyLoss(weights)
    else:
        loss = nn.CrossEntropyLoss()
    return loss


def load_files_from_dataset(dataset_name):
    file_path = find_path(f'config/dataset/**/{dataset_name}.yaml', True)
    with open(file_path, 'r') as stream:
        return np.array(sorted(yaml.safe_load(stream)))


def load_model_locally(uuid, model_name='rvce', device=None) -> Tuple[Any, Config]:
    if device is None:
        device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

    config = load_config_locally(uuid)
    weights = torch.load(f'outputs/{uuid}/weights/{model_name}.pth', device)
    model = get_model(config).to(device)
    model.load_state_dict(weights)
    model.eval()

    return model, config