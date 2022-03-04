from torchvision.models import resnet18
import torch.nn as nn


class ResNet18(nn.Module):
    def __init__(self, config, in_channels=1, pretrained=False):
        super(ResNet18, self).__init__()
        self.num_classes = config.num_classes
        self.model = resnet18(pretrained=pretrained)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.heads = nn.ModuleDict()
        for head in config.heads:
            self.add_head(head)

    def add_head(self, name):
        self.heads.add_module(name, nn.Linear(self.in_features, self.num_classes))

    def forward(self, x):
        x = self.model(x)
        heads = {name: head(x) for name, head in self.heads.items()}
        return heads


if __name__ == "__main__":
    model = ResNet18()
    print(model)