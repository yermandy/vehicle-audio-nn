from model.kullback import *

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from datetime import datetime


def run(audio_file, labels_file, n_epochs=150, cuda=0):

    device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')

    print(f'Running on {device}')

    trn_dataset = VehicleDataset(audio_file, labels_file, start_time=60, end_time=25 * 60, use_offset=True)
    trn_loader = DataLoader(trn_dataset, batch_size=64, shuffle=True)

    val_dataset = VehicleDataset(audio_file, labels_file, start_time=25 * 60, seed=0)
    val_loader = DataLoader(val_dataset, batch_size=64)

    model = ResNet18(num_classes=100).to(device)

    loss = nn.KLDivLoss(reduction='sum')

    optim = Adam(model.parameters(), lr=0.0001)

    loop = tqdm(range(n_epochs))

    val_loss_best = float('inf')

    uuid = int(datetime.now().timestamp())

    # '''

    for epoch in loop:

        trn_loss = 0
        val_loss = 0

        model.train()
        for tensor, target in trn_loader:
            
            tensor = tensor.to(device)
            target = target.to(device)
            
            scores: torch.Tensor = model(tensor)

            loss_val = loss(scores.log_softmax(1), target.softmax(1))

            # preds = scores.argmax(1)

            trn_loss += loss_val.detach().item()
            # trn_correct += (target == preds).sum()

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
                
                loss_val = loss(scores.log_softmax(1), target.softmax(1))
                val_loss += loss_val.detach().item()

        if val_loss < val_loss_best:
            val_loss_best = val_loss
            torch.save(model.state_dict(), f'weights/kullback/model_{uuid}.pth')

        loop.set_description(f'trn loss {trn_loss:.6f} | val loss {val_loss:.6f} | val best {val_loss_best:.6f}')
        # loop.set_description(f'loss {trn_loss.item():.4f}')

    # '''

if __name__ == "__main__":

    audio_file = 'data/audio/20190819-Kutna Hora-L4-out-MVI_0040.wav'
    labels_file = 'data/labels/20190819-Kutna Hora-L4-out-MVI_0040.txt'

    for i in range(5):
        run(audio_file, labels_file)