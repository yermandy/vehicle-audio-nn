import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool1d(2, stride=1, padding=1),
        )

    def forward(self, x):
        return self.block(x)


class WaveCNN(nn.Module):
    def __init__(self, config):
        super(WaveCNN, self).__init__()
        self.in_features = 128
        self.num_classes = config.num_classes

        self.net = nn.Sequential(
            nn.Identity(),
            BasicBlock(1, 128, 512, 8),
            BasicBlock(128, 128, 64, 4),
            BasicBlock(128, 128, 32, 2),
            BasicBlock(128, 128, 16, 2),
            BasicBlock(128, 128, 8, 2),
            BasicBlock(128, 128, 4, 2),
            BasicBlock(128, 128, 4, 1),
            BasicBlock(128, 128, 4, 1),
            BasicBlock(128, 128, 4, 1),
            BasicBlock(128, 128, 4, 1),
            BasicBlock(128, 128, 4, 1),
            nn.AdaptiveAvgPool1d(1),
        )
        # self.linear = nn.Linear(self.in_features, self.num_classes)
        self.heads = nn.ModuleDict()
        for head in config.heads:
            self.add_head(head)

    def add_head(self, name):
        self.heads.add_module(name, nn.Linear(self.in_features, self.num_classes))

    def forward(self, x):
        x = self.net(x)
        x = x.reshape(x.shape[0], -1)
        heads = {name: head(x) for name, head in self.heads.items()}
        return heads
