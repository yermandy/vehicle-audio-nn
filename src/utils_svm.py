from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn

from .datapool import DataPool
from .dataset import VehicleDataset
from .config import Config
from .constants import Part

from torch.utils.data import DataLoader
from tqdm import tqdm


def get_XY(files: list[str], config: Config, model: nn.Module, head: str):
    X = []
    Y = []

    datapool = DataPool(files, config)

    dataset = VehicleDataset(datapool, part=Part.WHOLE, config=config)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    device = next(model.parameters()).device
    dtype = torch.float16 if config.cuda >= 0 and config.fp16 else torch.float32
    device_type = "cuda" if config.cuda >= 0 else "cpu"

    model.eval()
    with torch.no_grad():
        for tensor, labels in tqdm(loader, leave=True):

            with torch.autocast(device_type=device_type, dtype=dtype):
                tensor = tensor.to(device)

            X.extend(model.features(tensor).detach().cpu().numpy())
            Y.extend(labels[head])

    X = np.array(X)
    Y = np.array(Y)

    return X, Y


def get_XY_heads(files: list[str], config: Config, model: nn.Module):
    X = []
    Y = defaultdict(list)

    datapool = DataPool(files, config)

    dataset = VehicleDataset(datapool, part=Part.WHOLE, config=config)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    device = next(model.parameters()).device
    dtype = torch.float16 if config.cuda >= 0 and config.fp16 else torch.float32
    device_type = "cuda" if config.cuda >= 0 else "cpu"

    model.eval()
    with torch.no_grad():
        for tensor, labels in tqdm(loader, leave=True):

            with torch.autocast(device_type=device_type, dtype=dtype):
                tensor = tensor.to(device)

            X.extend(model.features(tensor).detach().cpu().numpy())
            for head in labels:
                Y[head].extend(labels[head])

    X = np.array(X)
    Y = {head: np.array(Y[head]) for head in Y}

    return X, Y
