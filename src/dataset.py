from torch.utils.data import Dataset
from src import *

from easydict import EasyDict

class VehicleDataset(Dataset):

    def __init__(self,
            datapool: DataPool,
            part: bool = Part.WHOLE,
            config: EasyDict = EasyDict(),
            offset: int = 0):
            
        self.datapool = datapool
        self.config = config
        self.window_length = config.window_length
        self.part = part
        self.create_with_offset(offset)
        self.transform = create_transformation(config, part)
    
    def create_with_offset(self, offset):
        self.samples, self.labels = create_dataset_from_files(self.datapool, self.part, offset)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        label = {}
        for k, v in self.labels.items():
            label[k] = v[index]
        features = self.transform(sample)
        return features, label
