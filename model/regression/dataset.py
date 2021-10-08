import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from src import *
from typing import List

from easydict import EasyDict

class VehicleDataset(Dataset):

    def __init__(self,
            files: List[str], 
            from_time: float = 0,
            till_time: float = int(1e8),
            seed: int = 42,
            params: EasyDict = EasyDict(),
            n_samples: int = 5000):

        self.params = params
        self.window_length = get_window_length(params)
        self.samples, self.labels = create_dataset_from_files(
            files, 
            window_length=self.window_length,
            n_samples=n_samples,
            seed=seed,
            from_time=from_time,
            till_time=till_time)
        self.transform = create_transformation(params)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        features = self.transform(sample)
        return features, label
