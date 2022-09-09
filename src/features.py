from .dataset import VehicleDataset
from .utils import *
from .transformation import create_transformation

from torch.utils.data import DataLoader

from tqdm import tqdm


def extract_features(video: Video, model: nn.Module):
    video.config.set_window_length(6)
    video.config.set_nn_hop_length(3)

    dataset = VehicleDataset(video, part=Part.WHOLE, config=video.config)
    loader = DataLoader(
        dataset,
        batch_size=video.config.batch_size,
        num_workers=video.config.num_workers,
    )

    X = []

    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for tensor, labels in tqdm(loader, leave=True):
            tensor = tensor.to(device)

            X.extend(model.features(tensor).detach().cpu().numpy())

    X = np.array(X)[:-1]

    return X


def extract_labels(video: Video, head_name="n_counts"):
    video.config.set_window_length(3)
    video.config.set_nn_hop_length(3)

    dataset = VehicleDataset(video, part=Part.WHOLE, config=video.config)
    return np.array(dataset.labels[head_name])
