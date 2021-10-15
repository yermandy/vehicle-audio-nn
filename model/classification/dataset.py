from torch.utils.data import Dataset
from src import *

from easydict import EasyDict

class VehicleDataset(Dataset):

    def __init__(self,
            datapool: DataPool,
            is_trn: bool = True,
            seed: int = 42,
            params: EasyDict = EasyDict(),
            n_samples: int = 100,
            offset: int = 0):
            
        self.datapool = datapool
        self.params = params
        self.window_length = get_window_length(params)
        self.n_samples = n_samples
        self.seed = seed
        self.is_trn = is_trn
        self.samples, self.labels = create_dataset_from_files(
            datapool, 
            window_length=self.window_length,
            n_samples=n_samples,
            seed=seed,
            is_trn=is_trn,
            offset=offset)
        self.transform = create_transformation(params)
    
    def set_offset(self, offset):
        self.samples, self.labels = create_dataset_from_files(
            datapool=self.datapool, 
            window_length=self.window_length,
            n_samples=self.n_samples,
            seed=self.seed,
            is_trn=self.is_trn,
            offset=offset)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        features = self.transform(sample)
        return features, label
