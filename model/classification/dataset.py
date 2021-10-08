from torch.utils.data import Dataset
# from utils import *
from src import *

from easydict import EasyDict

class VehicleDataset(Dataset):

    def __init__(self,
            datapool: DataPool,
            is_trn: bool = True,
            seed: int = 42,
            params: EasyDict = EasyDict(),
            n_samples: int = 100):
            
        self.datapool = datapool
        self.params = params
        self.window_length = get_window_length(params)
        self.samples, self.labels = create_dataset_from_files(
            datapool, 
            window_length=self.window_length,
            n_samples=n_samples,
            seed=seed,
            is_trn=is_trn)
        self.transform = create_transformation(params)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        features = self.transform(sample)
        return features, label
