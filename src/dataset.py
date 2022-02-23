from torch.utils.data import Dataset
from src import *

from easydict import EasyDict

class VehicleDataset(Dataset):

    def __init__(self,
            datapool: DataPool,
            is_trn: bool = True,
            seed: int = 42,
            config: EasyDict = EasyDict(),
            n_samples: int = 100,
            offset: int = 0):
            
        self.datapool = datapool
        self.config = config
        self.window_length = config.window_length
        self.n_samples = n_samples
        self.seed = seed
        self.is_trn = is_trn
        self.samples, self.labels, self.domains = create_dataset_from_files(
            datapool, 
            window_length=self.window_length,
            n_samples=n_samples,
            seed=seed,
            is_trn=is_trn,
            offset=offset)
        self.transform = create_transformation(config, is_trn)
    
    def set_offset(self, offset):
        self.samples, self.labels, self.domains = create_dataset_from_files(
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
        domain = self.domains[index]
        features = self.transform(sample)
        return features, label, domain