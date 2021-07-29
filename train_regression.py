import os
import wandb
import math
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from easydict import EasyDict
from model.regression import *


def get_split_indices(params):
    n_features = math.ceil(params.sr * params.window_length / params.hop_length)
    n_features_in_sec = params.sr / params.hop_length
    n_features_in_frame = params.frame_length * n_features_in_sec
    n_half_features_in_frame = n_features_in_frame / 2
    centers = torch.linspace(n_half_features_in_frame, n_features - n_half_features_in_frame, steps=params.n_frames)
    split_indices = []
    for c in centers:
        from_idx = int(c - n_half_features_in_frame)
        to_idx = int(c + n_half_features_in_frame)
        split_indices.append([from_idx, to_idx])
    return split_indices


def run(audio_file, labels_file, n_epochs=500, cuda=0, window_length=5.0, frame_length=2.0, n_frames=3):
    
    # define parameters
    params = EasyDict()
    # length of one window in seconds
    params.window_length = window_length
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

    optim = Adam(model.parameters(), lr=0.0001)

    loop = tqdm(range(n_epochs))

    val_loss_best = float('inf')

    uuid = int(datetime.now().timestamp())

    config = wandb.config
    config.update(params)
    config.uuid = uuid

    wandb.run.name = str(uuid)

    split_indices = get_split_indices(params)

    for epoch in loop:

        trn_loss = 0
        val_loss = 0

        model.train()
        for tensor, target in trn_loader:

            tensor = tensor.to(device)
            target = target.to(device)

            n_samples = tensor.shape[0]

            Y = torch.zeros(n_samples).to(device)

            for from_idx, to_idx in split_indices:
                x = tensor[..., from_idx:to_idx]
                Y += model(x).squeeze()

            loss_value = (Y - target).abs().sum()

            trn_loss += loss_value.detach().item()

            optim.zero_grad()
            loss_value.backward()
            optim.step()

        wandb.log({"trn loss": trn_loss})

        trn_loader.dataset.split_signal()

        model.eval()
        with torch.no_grad():
            for tensor, target in val_loader:
                tensor = tensor.to(device)
                target = target.to(device)

                n_samples = tensor.shape[0]

                Y = torch.zeros(n_samples).to(device)

                for from_idx, to_idx in split_indices:
                    x = tensor[..., from_idx:to_idx]
                    Y += model(x).squeeze()

                loss_value = (Y - target).abs().sum()

                val_loss += loss_value.detach().item()

        wandb.log({"val loss": val_loss})

        if val_loss < val_loss_best:
            val_loss_best = val_loss
            torch.save(model.state_dict(), f'weights/regression/model_{uuid}.pth')

        wandb.log({"val loss best": val_loss_best})

        torch.save(model.state_dict(), f'weights/regression/model_{uuid}_last.pth')

        loop.set_description(f'trn loss {trn_loss:.2f} | val loss {val_loss:.2f} | best loss {val_loss_best:.2f}')


if __name__ == "__main__":

    os.makedirs('weights/regression', exist_ok=True)

    audio_file = 'data/audio/20190819-Kutna Hora-L4-out-MVI_0040.wav'
    labels_file = 'data/labels/20190819-Kutna Hora-L4-out-MVI_0040.txt'

    for window_length in [5.0, 6.0, 7.0]:
        for frame_length in [1.5, 2.0, 2.5, 3.0]:
            for n_frames in [3, 4, 5]:
                print('window_length:', window_length)
                print('frame_length:', frame_length)
                print('n_frames:', n_frames)

                wandb_run = wandb.init(project='vehicle-audio-nn', entity='yermandy')

                run(audio_file, labels_file, window_length=window_length, frame_length=frame_length, n_frames=n_frames)

                wandb_run.finish()
