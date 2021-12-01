from torchvision.models import resnet18
import torch.nn as nn


class ResNet18(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, pretrained=False):
        super(ResNet18, self).__init__()
        self.model = resnet18(pretrained=pretrained)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = ResNet18()
    print(model)