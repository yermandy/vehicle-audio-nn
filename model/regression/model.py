from torchvision.models import resnet18
from torch.nn import Conv2d, Module


class ResNet18(Module):
    def __init__(self, in_channels=1):
        super(ResNet18, self).__init__()
        self.model = resnet18(num_classes=1)
        self.model.conv1 = Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.model(x).abs()


if __name__ == "__main__":
    model = ResNet18()
    print(model)
    from torchsummary import summary
    summary(model, (1, 224, 224))