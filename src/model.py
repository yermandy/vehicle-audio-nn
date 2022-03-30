from torchvision.models import resnet18
import torch.nn as nn
from .config import Config


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


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 4, padding='same'), 
            nn.BatchNorm1d(out_channels),
            nn.ReLU(True),
            nn.MaxPool1d(4, 2, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class Mean(nn.Module):
    def __init__(self):
        super(Mean, self).__init__()

    def forward(self, x):
        return x.mean(-1)



class WaveCNN(nn.Module):
    def __init__(self, config: Config):
        super(WaveCNN, self).__init__()
        self.in_features = 128
        self.num_classes = config.num_classes
        self.net = nn.Sequential(
            BasicBlock(1, 16),
            BasicBlock(16, 32),
            BasicBlock(32, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 128),
        )
        self.resnet = ResNet18(config)
        # self.heads = nn.ModuleDict()
        # for head in config.heads:
        #     self.add_head(head)
        
    def add_head(self, name):
        self.heads.add_module(name, nn.Linear(self.in_features, self.num_classes))
    
    def forward(self, x):
        x = self.net(x).unsqueeze(1)
        return self.resnet(x)
        heads = {name: head(x) for name, head in self.heads.items()}
        return heads


if __name__ == "__main__":
    model = ResNet18()
    print(model)