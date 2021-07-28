from model.regression import *

import os
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from datetime import datetime


def run(audio_file, labels_file, n_epochs=500, cuda=0):

    device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')

    print(f'Running on {device}')

    trn_dataset = VehicleDataset(audio_file, labels_file, start_time=60, end_time=25 * 60, use_offset=True)
    trn_loader = DataLoader(trn_dataset, batch_size=64, shuffle=True)

    val_dataset = VehicleDataset(audio_file, labels_file, start_time=25 * 60, seed=0)
    val_loader = DataLoader(val_dataset, batch_size=64)

    model = ResNet18().to(device)

    optim = Adam(model.parameters(), lr=0.0001)

    loop = tqdm(range(n_epochs))

    val_loss_best = float('inf')

    uuid = int(datetime.now().timestamp())

    n_frames = 3

    for epoch in loop:

        trn_loss = 0
        val_loss = 0

        model.train()
        for tensor, target in trn_loader:

            tensor = tensor.to(device)
            target = target.to(device)

            X = tensor.split(tensor.shape[3] // n_frames + 1, dim=3)

            Y = torch.zeros(tensor.shape[0]).to(device)

            for x in X:
                Y += model(x).squeeze()

            loss_value = (Y - target).abs().sum()

            trn_loss += loss_value.detach().item()

            optim.zero_grad()
            loss_value.backward()
            optim.step()

        trn_loader.dataset.split_signal()

        model.eval()
        with torch.no_grad():
            for tensor, target in val_loader:
                tensor = tensor.to(device)
                target = target.to(device)

                X = tensor.split(tensor.shape[3] // n_frames + 1, dim=3)

                Y = torch.zeros(tensor.shape[0]).to(device)

                for x in X:
                    Y += model(x).squeeze()

                loss_value = (Y - target).abs().sum()

                val_loss += loss_value.detach().item()

        if val_loss < val_loss_best:
            val_loss_best = val_loss
            torch.save(model.state_dict(), f'weights/regression/model_{uuid}.pth')
        
        torch.save(model.state_dict(), f'weights/regression/model_{uuid}_last.pth')

        loop.set_description(f'trn loss {trn_loss:.2f} | val loss {val_loss:.2f} | best loss {val_loss_best:.2f}')


if __name__ == "__main__":

    os.makedirs('weights/regression', exist_ok=True)

    audio_file = 'data/audio/20190819-Kutna Hora-L4-out-MVI_0040.wav'
    labels_file = 'data/labels/20190819-Kutna Hora-L4-out-MVI_0040.txt'

    for i in range(3):
        run(audio_file, labels_file)
