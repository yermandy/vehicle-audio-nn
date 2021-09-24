import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from ..utils import *
from copy import deepcopy

from easydict import EasyDict

class VehicleDataset(Dataset):

    def __init__(self,
            signal: torch.Tensor,
            events: np.array,
            start_time: float = 0,
            end_time: float = int(1e8),
            seed: int = 42,
            use_offset: bool = False,
            params: EasyDict = EasyDict(),
            n_samples: int = 5000):

        self.start_time = start_time
        self.end_time = end_time
        self.use_offset = use_offset
        self.seed = seed
        self.params = params

        self.signal = deepcopy(signal)
        self.events = deepcopy(events)
        self.window_length = get_window_length(params)

        self.samples, self.labels = create_dataset_uniformly(signal, params.sr, events, start_time, end_time, self.window_length, n_samples, seed)
        
        self.transform = create_transformation(params)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        features = self.transform(sample)
        return features, label


class VehicleValidationDataset(Dataset):
    
    def __init__(self, signal: torch.Tensor, params: EasyDict, transform):
        self.signal = deepcopy(signal)
        self.params = deepcopy(params)
        self.transform = transform
        
    def __len__(self):
        return self.params.n_hops

    def __getitem__(self, index):
        start = index * self.params.n_samples_in_nn_hop
        end = start + self.params.n_samples_in_frame
        x = self.params.signal[start: end]
        x = self.transform(x)
        return x