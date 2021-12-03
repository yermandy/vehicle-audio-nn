import torch
import torchaudio
import numpy as np
from collections import defaultdict, Counter
import pickle

def time_to_sec(time):
    h, m, s = map(float, time.split(':'))
    sec = h * 3600 + m * 60 + s
    return sec


def load_csv(name, folder='data/csv/*.csv', preprocess=True):
    import glob
    batch = []
    for file in glob.glob(folder):
        if file.find(name) != -1:
            table = np.genfromtxt(file, dtype=str, delimiter=';', skip_header=1)
            batch.append(table)
    batch = np.concatenate(batch)
    if preprocess:
        batch = preprocess_csv(batch)
    return batch


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

    start_time_column_id = 8
    end_time_column_id = 9
    best_detection_frame_time_column_id = 14
    views_column_id = 23
    
    for row in csv:
        plate_id = row[1]
        start_time = row[start_time_column_id]
        end_time = row[end_time_column_id]
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
            
            most_common_view = Counter(item for item in traking_rows[:, views_column_id]).most_common(1)[0][0]
            
            modified_row = np.full_like(traking_rows[0], '')
            
            start_times = traking_rows[:, start_time_column_id]
            end_times = traking_rows[:, end_time_column_id]
            best_detection_frame_times = traking_rows[:, best_detection_frame_time_column_id]
            
            times = np.concatenate([start_times, end_times])
            times = np.sort(times)
            times = [t for t in times if t != '']
            
            modified_row[0] = tracking_uuid
            modified_row[1] = key
            tracking_uuid += 1
            modified_row[best_detection_frame_time_column_id] = best_detection_frame_times[-1]
            modified_row[start_time_column_id] = times[0]
            modified_row[end_time_column_id] = times[-1]
            modified_row[views_column_id] = most_common_view
            rows.append(modified_row)
    rows = np.array(rows)
    indices = np.argsort(rows[:, start_time_column_id])
    rows = rows[indices]
    return rows


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
    return np.loadtxt(file)


def load_intervals_and_n_events(file):
    events = load_events(f'data/labels/{file}.MP4.txt')
    intervals = load_intervals(f'data/intervals/{file}.MP4.txt')

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
    return load_column(csv, 23)


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


def load_model_wandb(uuid, wandb_entity, wandb_project, model_name='mae', classification=True):
    import wandb
    from easydict import EasyDict
    if classification:
        from model.classification import ResNet18

    api = wandb.Api()
    runs = api.runs(f'{wandb_entity}/{wandb_project}', per_page=5000, order='config.uuid')

    for run in runs: 
        if run.name == str(uuid):
            config = EasyDict(run.config)
            break
    
    device = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')
    weights = torch.load(f'outputs/{uuid}/weights/{model_name}.pth', device)
    num_classes = len(weights['model.fc.bias'])
    model = ResNet18(num_classes=num_classes).to(device)
    model.load_state_dict(weights)
    return model, config


def load_model_locally(uuid, model_name='mae', classification=True):
    if classification:
        from model.classification import ResNet18

    device = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')
    weights = torch.load(f'outputs/{uuid}/weights/{model_name}.pth', device)
    num_classes = len(weights['model.fc.bias'])
    model = ResNet18(num_classes=num_classes).to(device)
    model.load_state_dict(weights)

    with open(f'outputs/{uuid}/config.pickle', 'rb') as f:
        config = pickle.load(f)

    return model, config