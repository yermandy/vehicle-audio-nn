from torchvision.models import resnet18
from torch.nn import Conv2d, Module


class ResNet18(Module):
    def __init__(self, in_channels=1):
        super(ResNet18, self).__init__()
        self.model = resnet18(num_classes=1)
        self.model.conv1 = Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.model(x).abs()
        return x


if __name__ == "__main__":
    model = ResNet18()
    print(model)
    from torchsummary import summary
    summary(model, (1, 224, 224))


from torchvision.models.mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small
import torch.nn as nn


def replace_in_channels(in_channels, conv: nn.Conv2d):
    return nn.Conv2d(in_channels, conv.out_channels,
        conv.kernel_size, conv.stride, conv.padding,
        conv.dilation, conv.groups, conv.bias, conv.padding_mode)


class MobileNetV3S(Module):
    def __init__(self):
        super(MobileNetV3S, self).__init__()
        model = mobilenet_v3_small(num_classes=1)
        conv: nn.Conv2d = model.features[0][0]
        model.features[0][0] = replace_in_channels(1, conv)
        self.model = model

    def forward(self, x):
        x = self.model(x).abs()
        return x


class MobileNetV3L(Module):
    def __init__(self):
        super(MobileNetV3L, self).__init__()
        model = mobilenet_v3_large(num_classes=1)
        conv: nn.Conv2d = model.features[0][0]
        model.features[0][0] = replace_in_channels(1, conv)
        self.model = model

    def forward(self, x):
        x = self.model(x).abs()
        return x