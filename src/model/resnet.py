import pyrootutils.root

from torchvision.models import resnet18, resnet34, resnet50
import torch.nn as nn
from src.config import Config


class ResNet(nn.Module):
    def __init__(self, model_class, config: Config, in_channels=1, pretrained=False):
        super(ResNet, self).__init__()
        self.num_classes = config.num_classes
        self.model = model_class(pretrained=pretrained)
        self.model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.heads = nn.ModuleDict()
        for head in config.heads:
            self.add_head(head)

    def add_head(self, name):
        self.heads.add_module(name, nn.Linear(self.in_features, self.num_classes))

    def forward(self, x):
        x = self.features(x)
        heads = {name: head(x) for name, head in self.heads.items()}
        return heads

    def features(self, x):
        return self.model(x)

    def forward_single_head(self, x):
        x = self.model(x)
        return self.heads["n_counts"](x)


class ResNet18(ResNet):
    def __init__(self, config, in_channels=1, pretrained=False):
        super().__init__(resnet18, config, in_channels, pretrained)


class ResNet34(ResNet):
    def __init__(self, config, in_channels=1, pretrained=False):
        super().__init__(resnet34, config, in_channels, pretrained)


class ResNet50(ResNet):
    def __init__(self, config, in_channels=1, pretrained=False):
        super().__init__(resnet50, config, in_channels, pretrained)


if __name__ == "__main__":

    config = Config()
    print(config)
    model = ResNet18(config)

    print(model)
