from .dataset import VehicleDataset
from .utils import *
from .transformation import create_transformation

from torch.utils.data import DataLoader

from tqdm import tqdm

def extract_features(video: Video, model: nn.Module, head_name="n_counts"):

    dataset = VehicleDataset(video, part=Part.WHOLE, config=video.config)
    loader = DataLoader(
        dataset,
        batch_size=video.config.batch_size,
        num_workers=video.config.num_workers,
    )

    X = []
    Y = []

    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for tensor, labels in tqdm(loader, leave=True):
            tensor = tensor.to(device)

            X.extend(model.features(tensor).detach().cpu().numpy())
            Y.extend(labels[head_name])

    X = np.array(X)
    Y = np.array(Y)

    return X, Y
