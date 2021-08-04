import os
import easydict
import wandb
import math
import torch
import torchaudio
import numpy as np
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from easydict import EasyDict
from model.regression import *


def load_audio(audio_file):
    signal, _ = torchaudio.load(audio_file)
    signal = signal.mean(0)
    return signal


def load_events(events_file):
    return np.loadtxt(events_file)


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
    def closest(array, value): return np.abs(array - value).argmin()
    cumstep = np.zeros_like(T)
    for e in E:
        idx = closest(T, e)
        cumstep[idx] += 1
    return np.cumsum(cumstep)


def get_diff(results, params):
    cumsum = np.cumsum(results)
    cumstep = get_cumstep(params.time, params.events)
    return np.abs(cumsum - cumstep).mean()


def get_additional_params(signal, events, params, start_time=0, end_time=0):
    signal = signal[start_time * params.sr: end_time * params.sr]

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
    
    # take events that are in interval
    events = events[(events >= start_time) & (events < end_time)]
    
    # create time axis for prediction visualization
    nn_hop_length_half = params.nn_hop_length // 2
    time = np.linspace(start_time + nn_hop_length_half, end_time - nn_hop_length_half, n_hops)

    additional = EasyDict()

    additional.signal = signal
    additional.n_hops = n_hops
    additional.events = events
    additional.time = time
    additional.n_samples_in_nn_hop = n_samples_in_nn_hop
    additional.n_samples_in_frame = n_samples_in_frame
    additional.n_features_in_nn_hop = n_features_in_nn_hop
    additional.n_features_in_frame = n_features_in_frame
    additional.n_features_in_interval = n_features_in_interval

    return additional


def validate(model, dataset, params, tqdm=lambda x: x):
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
            y = model(x).item()
            results.append(y)

    results = np.array(results)
    return results


def run(audio_file, labels_file, nn_hop_length=5.0, frame_length=2.0, n_frames=3):
    uuid=int(datetime.now().timestamp())

    TRN_FROM_TIME = 1 * 60
    TRN_TILL_TIME = 25 * 60
    VAL_FROM_TIME = 25 * 60
    VAL_TILL_TIME = 34 * 60

    cuda = 0
    n_epochs = 500
    batch_size = 64

    # define parameters
    params = EasyDict()
    # hop length between nn inputs in features
    params.nn_hop_length = nn_hop_length
    # length of one frame in seconds
    params.frame_length = frame_length
    # number of frames in one window
    params.n_frames = n_frames
    # length of one feature in samples
    params.n_fft = 1024
    # number of mel features
    params.n_mels = 64
    # number of mfcc features
    params.n_mfcc = 8
    # sampling rate
    params.sr = 44100
    # hop length between samples for feature extractor
    params.hop_length = 128
    # length of one window in seconds
    params.window_length = params.nn_hop_length * (params.n_frames - 1) + params.frame_length

    device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')

    print(f'Running on {device}')

    signal = load_audio(audio_file)
    events = load_events(labels_file)

    trn_dataset = VehicleDataset(
        signal,
        events,
        start_time=TRN_FROM_TIME,
        end_time=TRN_TILL_TIME,
        use_offset=True,
        params=params
    )

    trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = VehicleDataset(
        signal,
        events,
        start_time=VAL_FROM_TIME,
        end_time=VAL_TILL_TIME,
        seed=0,
        params=params
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = ResNet18().to(device)

    optim = AdamW(model.parameters(), lr=0.0001)

    val_loss_best = float('inf')
    val_smallest_diff = float('inf')

    config = wandb.config
    config.update(params)
    config.uuid = uuid
    config.batch_size = batch_size

    wandb.run.name = str(uuid)

    split_indices = get_split_indices(params)

    params.trn = get_additional_params(
        signal, events, params, start_time=TRN_FROM_TIME, end_time=TRN_TILL_TIME
    )
    
    params.val = get_additional_params(
        signal, events, params, start_time=VAL_FROM_TIME, end_time=VAL_TILL_TIME
    )

    training_loop = tqdm(range(n_epochs))
    for _ in training_loop:

        # training
        trn_loss = 0
        n_processed = 0

        model.train()
        for tensor, target in trn_loader:

            tensor = tensor.to(device)
            target = target.to(device)

            n_samples = tensor.shape[0]
            n_processed += n_samples

            Y = torch.zeros(n_samples).to(device)

            for start, end in split_indices:
                x = tensor[..., start: end]
                Y += model(x).squeeze()

            loss_value = (Y - target).abs().sum()

            trn_loss += loss_value.detach().item()

            optim.zero_grad()
            loss_value.backward()
            optim.step()

        trn_loss /= len(trn_dataset) * len(split_indices)
        # creates a new offset
        trn_loader.dataset.split_signal()

        # validation
        val_loss = 0
        n_processed = 0

        model.eval()
        with torch.no_grad():
            for tensor, target in val_loader:
                tensor = tensor.to(device)
                target = target.to(device)

                n_samples = tensor.shape[0]
                n_processed += n_samples * len(split_indices)

                Y = torch.zeros(n_samples).to(device)

                for start, end in split_indices:
                    x = tensor[..., start: end]
                    Y += model(x).squeeze()

                loss_value = (Y - target).abs().sum()

                val_loss += loss_value.detach().item()

        val_loss /= len(trn_dataset) * len(split_indices)

        trn_results = validate(model, trn_dataset, params.trn)
        trn_diff = get_diff(trn_results, params.trn)

        val_results = validate(model, val_dataset, params.val)
        val_diff = get_diff(val_results, params.val)

        if val_diff < val_smallest_diff:
            val_smallest_diff = val_diff
            torch.save(model.state_dict(), f'weights/regression/model_{uuid}_diff.pth')

        if val_loss < val_loss_best:
            val_loss_best = val_loss
            torch.save(model.state_dict(), f'weights/regression/model_{uuid}_loss.pth')

        wandb.log({
            "trn loss": trn_loss,
            "val loss": val_loss,
            "trn diff": trn_diff,
            "val diff": val_diff,
            "val loss best": val_loss_best,
            "val smallest diff": val_smallest_diff
        })

        training_loop.set_description(
            f'trn loss {trn_loss:.2f} | val loss {val_loss:.2f} | best loss {val_loss_best:.2f}')


if __name__ == "__main__":

    os.makedirs('weights/regression', exist_ok=True)

    audio_file = 'data/audio/20190819-Kutna Hora-L4-out-MVI_0040.wav'
    labels_file = 'data/labels/20190819-Kutna Hora-L4-out-MVI_0040.txt'

    # '''
    # for nn_hop_length in [1.0, 2.0, 3.0]:
    for nn_hop_length in [2.0, 3.0]:
        for frame_length in [1.0, 2.0, 3.0]:
            for n_frames in [3, 5, 7]:
                if nn_hop_length > frame_length:
                    continue

                print('nn_hop_length:', nn_hop_length)
                print('frame_length:', frame_length)
                print('n_frames:', n_frames)

                wandb_run = wandb.init(project='vehicle-audio-nn', entity='yermandy', tags=['grid search'])

                run(audio_file, labels_file, nn_hop_length=nn_hop_length,
                    frame_length=frame_length, n_frames=n_frames)

                wandb_run.finish()
    # '''

    '''
    wandb_run = wandb.init(project='test', entity='yermandy', tags=['test'])
    run(audio_file, labels_file, nn_hop_length=2, frame_length=2, n_frames=3)
    wandb_run.finish()
    # '''
