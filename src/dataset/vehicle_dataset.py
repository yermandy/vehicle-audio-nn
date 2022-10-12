from torch.utils.data import Dataset
from src import *

from easydict import EasyDict


class VehicleDataset(Dataset):
    def __init__(
        self,
        data: DataPool,  # or Video
        part: Part = Part.WHOLE,
        config: Config = EasyDict(),
        offset: int = 0,
        is_trn: bool = False,
    ):

        self.data = data
        self.config = config
        self.window_length = config.window_length
        self.part = part
        self.create_with_offset(offset)
        self.transform = create_transformation(config, is_trn)

    def create_with_offset(self, offset):
        if isinstance(self.data, DataPool):
            self.samples, self.labels = create_dataset_from_datapool(
                self.data, self.part, offset
            )
        elif isinstance(self.data, Video):
            from_time, till_time = self.data.get_from_till_time(self.part)
            from_time = from_time + offset
            self.samples, self.labels = create_dataset_from_video(
                self.data, from_time=from_time, till_time=till_time
            )
        else:
            raise ValueError("Unknown data type")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        label = {}
        for k, v in self.labels.items():
            label[k] = v[index]
        features = self.transform(sample)
        return features, label
