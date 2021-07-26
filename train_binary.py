from numpy.random import seed
from model.binary import *

import os
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from datetime import datetime

def run(audio_file, labels_file, n_epochs=300, cuda=0):

    device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')

    print(f'Running on {device}')

    trn_dataset = VehicleDataset(audio_file, labels_file, start_time=60, end_time=25 * 60, use_offset=True)
    trn_loader = DataLoader(trn_dataset, batch_size=64, shuffle=True)

    val_dataset = VehicleDataset(audio_file, labels_file, start_time=25 * 60, seed=0)
    val_loader = DataLoader(val_dataset, batch_size=64)

    model = ResNet18().to(device)

    loss = nn.CrossEntropyLoss()

    optim = Adam(model.parameters(), lr=0.0001)

    loop = tqdm(range(n_epochs))

    val_correct_best = 0

    uuid = int(datetime.now().timestamp())

    for epoch in loop:

        trn_correct = 0
        val_correct = 0
        trn_loss = 0

        model.train()
        for tensor, target in trn_loader:
            
            tensor = tensor.to(device)
            target = target.to(device)
            
            scores = model(tensor)
            loss_val = loss(scores, target)

            preds = scores.argmax(1)

            trn_loss += loss_val.detach()
            trn_correct += (target == preds).sum()

            optim.zero_grad()
            loss_val.backward()
            optim.step()

        trn_loader.dataset.split_signal()
        
        model.eval()
        with torch.no_grad():
            for tensor, target in val_loader:
                tensor = tensor.to(device)
                target = target.to(device)

                scores = model(tensor)
                preds = scores.argmax(1)
                val_correct += (target == preds).sum()

        if val_correct > val_correct_best:
            val_correct_best = val_correct
            torch.save(model.state_dict(), f'weights/binary/model_{uuid}.pth')

        loop.set_description(f'loss {trn_loss.item():.4f} | trn acc {trn_correct / len(trn_dataset):.4f} | val acc {val_correct / len(val_dataset):.4f} | best acc {val_correct_best / len(val_dataset):.4f}')


if __name__ == "__main__":

    os.makedirs('weights/binary', exist_ok=True)

    audio_file = 'data/audio/20190819-Kutna Hora-L4-out-MVI_0040.wav'
    labels_file = 'data/labels/20190819-Kutna Hora-L4-out-MVI_0040.txt'

    for i in range(5):
        run(audio_file, labels_file)
