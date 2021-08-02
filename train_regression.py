import os
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
        idx = closest(T, e) + 1
        cumstep[idx] += 1
    return np.cumsum(cumstep)


def get_validation_data(audio_file, labels_file, params, start_time=25*60, end_time=34*60):
    signal, _ = torchaudio.load(audio_file)
    signal = signal.mean(0)

    events = np.loadtxt(labels_file)

    signal = signal[start_time * params.sr: end_time * params.sr]
    
    interval = end_time - start_time

    n_features_in_sec = params.sr / params.hop_length
    n_features_in_nn_hop = math.ceil(params.nn_hop_length * n_features_in_sec)
    n_features_in_frame = math.ceil(params.frame_length * n_features_in_sec)
    n_featues_in_interval = interval * n_features_in_sec
    
    # number of hops
    K = math.ceil((n_featues_in_interval - n_features_in_frame) / n_features_in_nn_hop)
    # take events that are in interval
    E = events[(events >= start_time) & (events < end_time)]
    # K time points between start_time and end_time
    T = np.linspace(start_time, end_time, K)

    params.signal = signal
    params.K = K 
    params.E = E 
    params.T = T
    params.n_features_in_nn_hop = n_features_in_nn_hop
    params.n_features_in_frame = n_features_in_frame

    return params
    

def validate(params, model, dataset, device):
    results = []
    X = dataset.transform(params.signal).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        for k in range(params.K):
            start = k * params.n_features_in_nn_hop
            end = k * params.n_features_in_nn_hop + params.n_features_in_frame
            x = X[..., start : end]
            x = x.to(device)
            y = model(x).item()
            results.append(y)
    results = np.array(results)

    cumsum = np.cumsum(results)
    cumstep = get_cumstep(params.T, params.E)
    diff = np.abs(cumsum - cumstep).mean()
    return diff



def run(audio_file, labels_file, n_epochs=500, cuda=0, nn_hop_length=5.0, frame_length=2.0, n_frames=3):
    
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

    trn_dataset = VehicleDataset(audio_file,
                                 labels_file,
                                 start_time=60,
                                 end_time=25 * 60,
                                 use_offset=True,
                                 params=params)

    trn_loader = DataLoader(trn_dataset, batch_size=64, shuffle=True)

    val_dataset = VehicleDataset(audio_file,
                                 labels_file,
                                 start_time=25 * 60,
                                 seed=0,
                                 params=params)

    val_loader = DataLoader(val_dataset, batch_size=64)

    model = ResNet18().to(device)

    optim = AdamW(model.parameters(), lr=0.0001)


    val_loss_best = float('inf')
    smallest_diff = float('inf')

    uuid = int(datetime.now().timestamp())

    config = wandb.config
    config.update(params)
    config.uuid = uuid

    wandb.run.name = str(uuid)

    split_indices = get_split_indices(params)

    params = get_validation_data(audio_file, labels_file, params)

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

            for from_idx, to_idx in split_indices:
                x = tensor[..., from_idx:to_idx]
                Y += model(x).squeeze()

            loss_value = (Y - target).abs().sum()

            trn_loss += loss_value.detach().item()

            optim.zero_grad()
            loss_value.backward()
            optim.step()

        trn_loss /= n_processed
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
                n_processed += n_samples

                Y = torch.zeros(n_samples).to(device)

                for from_idx, to_idx in split_indices:
                    x = tensor[..., from_idx:to_idx]
                    Y += model(x).squeeze()

                loss_value = (Y - target).abs().sum()

                val_loss += loss_value.detach().item()

        val_loss /= n_processed

        diff = validate(params, model, val_dataset, device)

        if diff < smallest_diff:
            smallest_diff = diff
            torch.save(model.state_dict(), f'weights/regression/model_{uuid}_diff.pth')

        if val_loss < val_loss_best:
            val_loss_best = val_loss
            torch.save(model.state_dict(), f'weights/regression/model_{uuid}_loss.pth')

        wandb.log({
            "trn loss": trn_loss,
            "val loss": val_loss,
            "diff": diff,
            "val loss best": val_loss_best,
            "smallest diff": smallest_diff
        })

        training_loop.set_description(f'trn loss {trn_loss:.2f} | val loss {val_loss:.2f} | best loss {val_loss_best:.2f}')


if __name__ == "__main__":

    os.makedirs('weights/regression', exist_ok=True)

    audio_file = 'data/audio/20190819-Kutna Hora-L4-out-MVI_0040.wav'
    labels_file = 'data/labels/20190819-Kutna Hora-L4-out-MVI_0040.txt'

    for nn_hop_length in [1.0, 2.0, 3.0]:
        for frame_length in [1.0, 2.0, 3.0]:
            for n_frames in [3, 4, 5, 6]:
                print('nn_hop_length:', nn_hop_length)
                print('frame_length:', frame_length)
                print('n_frames:', n_frames)

                wandb_run = wandb.init(project='vehicle-audio-nn', entity='yermandy')

                run(audio_file, labels_file, nn_hop_length=nn_hop_length, frame_length=frame_length, n_frames=n_frames)

                wandb_run.finish()