from distutils.command.config import config
from torchvision.models import resnet18
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

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


class ResNet1D(nn.Module):
    def __init__(self, config: Config):
        super(ResNet1D, self).__init__()
        n_feature_maps = 64
        
        # BLOCK 1
        kernel_size = 8
        self.padding1_x = nn.ConstantPad1d((int((kernel_size-1)/2), math.ceil((kernel_size-1)/2)), 0)
        self.conv1_x = torch.nn.Conv1d(in_channels=1, out_channels=n_feature_maps, kernel_size=kernel_size)
        self.bn1_x = nn.BatchNorm1d(num_features=n_feature_maps)
        self.relu1_x = nn.ReLU()

        kernel_size = 5
        self.padding1_y = nn.ConstantPad1d((int((kernel_size-1)/2), math.ceil((kernel_size-1)/2)), 0)
        self.conv1_y = torch.nn.Conv1d(in_channels=n_feature_maps, out_channels=n_feature_maps, kernel_size=kernel_size)
        self.bn1_y = nn.BatchNorm1d(num_features=n_feature_maps)
        self.relu1_y = nn.ReLU()

        kernel_size = 3
        self.padding1_z = nn.ConstantPad1d((int((kernel_size-1)/2), math.ceil((kernel_size-1)/2)), 0)
        self.conv1_z = torch.nn.Conv1d(in_channels=n_feature_maps, out_channels=n_feature_maps, kernel_size=kernel_size)
        self.bn1_z = nn.BatchNorm1d(num_features=n_feature_maps)

        self.conv1_sy = torch.nn.Conv1d(in_channels=1, out_channels=n_feature_maps, kernel_size=1)
        self.bn1_sy = nn.BatchNorm1d(num_features=n_feature_maps)

        # BLOCK 2
        kernel_size = 8
        self.padding2_x = nn.ConstantPad1d((int((kernel_size-1)/2), math.ceil((kernel_size-1)/2)), 0)
        self.conv2_x = torch.nn.Conv1d(in_channels=n_feature_maps, out_channels=n_feature_maps*2, kernel_size=kernel_size)
        self.bn2_x = nn.BatchNorm1d(num_features=n_feature_maps*2)
        self.relu2_x = nn.ReLU()

        kernel_size = 5
        self.padding2_y = nn.ConstantPad1d((int((kernel_size-1)/2), math.ceil((kernel_size-1)/2)), 0)
        self.conv2_y = torch.nn.Conv1d(in_channels=n_feature_maps*2, out_channels=n_feature_maps*2, kernel_size=kernel_size)
        self.bn2_y = nn.BatchNorm1d(num_features=n_feature_maps*2)
        self.relu2_y = nn.ReLU()

        kernel_size = 3
        self.padding2_z = nn.ConstantPad1d((int((kernel_size-1)/2), math.ceil((kernel_size-1)/2)), 0)
        self.conv2_z = torch.nn.Conv1d(in_channels=n_feature_maps*2, out_channels=n_feature_maps*2, kernel_size=kernel_size)
        self.bn2_z = nn.BatchNorm1d(num_features=n_feature_maps*2)

        self.conv2_sy = torch.nn.Conv1d(in_channels=n_feature_maps, out_channels=n_feature_maps*2, kernel_size=1)
        self.bn2_sy = nn.BatchNorm1d(num_features=n_feature_maps*2)

        # BLOCK 3
        kernel_size = 8
        self.padding3_x = nn.ConstantPad1d((int((kernel_size-1)/2), math.ceil((kernel_size-1)/2)), 0)
        self.conv3_x = torch.nn.Conv1d(in_channels=n_feature_maps*2, out_channels=n_feature_maps*2, kernel_size=kernel_size)
        self.bn3_x = nn.BatchNorm1d(num_features=n_feature_maps*2)
        self.relu3_x = nn.ReLU()

        kernel_size = 5
        self.padding3_y = nn.ConstantPad1d((int((kernel_size-1)/2), math.ceil((kernel_size-1)/2)), 0)
        self.conv3_y = torch.nn.Conv1d(in_channels=n_feature_maps*2, out_channels=n_feature_maps*2, kernel_size=kernel_size)
        self.bn3_y = nn.BatchNorm1d(num_features=n_feature_maps*2)
        self.relu3_y = nn.ReLU()

        kernel_size = 3
        self.padding3_z = nn.ConstantPad1d((int((kernel_size-1)/2), math.ceil((kernel_size-1)/2)), 0)
        self.conv3_z = torch.nn.Conv1d(in_channels=n_feature_maps*2, out_channels=n_feature_maps*2, kernel_size=kernel_size)
        self.bn3_z = nn.BatchNorm1d(num_features=n_feature_maps*2)

        self.bn3_sy = nn.BatchNorm1d(num_features=n_feature_maps * 2)

        self.averagepool = nn.AvgPool1d(kernel_size = config.n_samples_in_window)
        # self.hidden = nn.Linear(n_feature_maps * 2, config.num_classes)

        self.in_features = n_feature_maps * 2
        self.num_classes = config.num_classes

        self.heads = nn.ModuleDict()
        for head in config.heads:
            self.add_head(head)

    def add_head(self, name):
        self.heads.add_module(name, nn.Linear(self.in_features, self.num_classes))

    def forward(self, X):
        #block1
        temp1 = self.padding1_x(X)
        temp1 = self.conv1_x(temp1)
        temp1 = self.bn1_x(temp1)
        temp1 = self.relu1_x(temp1)

        temp1 = self.padding1_y(temp1)
        temp1 = self.conv1_y(temp1)
        temp1 = self.bn1_y(temp1)
        temp1 = self.relu1_y(temp1)

        temp1 = self.padding1_z(temp1)
        temp1 = self.conv1_z(temp1)
        temp1 = self.bn1_z(temp1)

        shot_cut_X = self.conv1_sy(X)
        shot_cut_X = self.bn1_sy(shot_cut_X)

        block1 = shot_cut_X + temp1
        block1 = F.relu(block1)

        # block2
        temp2 = self.padding2_x(block1)
        temp2 = self.conv2_x(temp2)
        temp2 = self.bn2_x(temp2)
        temp2 = self.relu2_x(temp2)

        temp2 = self.padding2_y(temp2)
        temp2 = self.conv2_y(temp2)
        temp2 = self.bn2_y(temp2)
        temp2 = self.relu2_y(temp2)

        temp2 = self.padding2_z(temp2)
        temp2 = self.conv2_z(temp2)
        temp2 = self.bn2_z(temp2)

        shot_cut_block1 = self.conv2_sy(block1)
        shot_cut_block1 = self.bn2_sy(shot_cut_block1)

        block2 = shot_cut_block1 + temp2
        block2 = F.relu(block2)

        # block3
        temp3 = self.padding3_x(block2)
        temp3 = self.conv3_x(temp3)
        temp3 = self.bn3_x(temp3)
        temp3 = self.relu3_x(temp3)

        temp3 = self.padding3_y(temp3)
        temp3 = self.conv3_y(temp3)
        temp3 = self.bn3_y(temp3)
        temp3 = self.relu3_y(temp3)

        temp3 = self.padding3_z(temp3)
        temp3 = self.conv3_z(temp3)
        temp3 = self.bn3_z(temp3)

        shot_cut_block2 = self.bn3_sy(block2)
        block3 = shot_cut_block2 + temp3
        block3 = F.relu(block3)

        X = self.averagepool(block3)
        X = X.squeeze_(-1)
        # X = self.hidden(X)

        heads = {name: head(X) for name, head in self.heads.items()}

        return heads


if __name__ == "__main__":
    config = Config()
    model = ResNet1D(config)

    X = torch.rand((config.batch_size, 1, config.n_samples_in_window))
    print(model(X))