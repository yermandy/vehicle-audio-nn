import torch
import torch.nn as nn
import torch.nn.functional as F


class RawNetBasicBlock(nn.Module):
    """
    Basic block for RawNet architectures.
    This block follows pre-activation[1].

    Arguments:
    downsample  : perform reduction in the sequential(time) domain
                  (different with shortcut)

    Reference:
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Identity Mappings in Deep Residual Networks. arXiv:1603.05027
    """

    def __init__(self, inplanes, planes, downsample=None):
        super(RawNetBasicBlock, self).__init__()
        self.downsample = downsample

        #####
        # core layers
        #####
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.conv1 = nn.Conv1d(
            inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.mp = nn.MaxPool1d(3)
        self.afms = AFMS(planes)
        self.lrelu = nn.LeakyReLU(0.3)

        #####
        # settings
        #####
        if inplanes != planes:  # if change in number of filters
            self.shortcut = nn.Sequential(
                nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
            )

    def forward(self, x):
        out = self.lrelu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(self.lrelu(self.bn2(out)))
        out = out + shortcut

        if self.downsample:
            out = self.mp(out)
        out = self.afms(out)

        return out


class AFMS(nn.Module):
    """
    Alpha-Feature map scaling, added to the output of each residual block[1,2].

    Reference:
    [1] RawNet2 : https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1011.pdf
    [2] AMFS    : https://www.koreascience.or.kr/article/JAKO202029757857763.page
    """

    def __init__(self, nb_dim):
        super(AFMS, self).__init__()
        self.alpha = nn.Parameter(torch.ones((nb_dim, 1)))
        self.fc = nn.Linear(nb_dim, nb_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        y = self.sig(self.fc(y)).view(x.size(0), x.size(1), -1)

        x = x + self.alpha
        x = x * y
        return x


class RawNet2(nn.Module):
    """
    Refactored RawNet2 architecture.

    Reference:
    [1] RawNet2 : https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1011.pdf
    """

    def __init__(
        self, block, layers, filters, num_classes=512, in_channels=1, **kwargs
    ):
        super(RawNet2, self).__init__()
        self.inplanes = filters[0]

        #####
        # first layers before residual blocks
        #####
        self.conv1 = nn.Conv1d(
            in_channels, filters[0], kernel_size=3, stride=3, padding=0
        )

        #####
        # residual blocks for frame-level representations
        #####
        self.residual_layers = nn.Sequential(
            *[self._make_layer(block, f, l) for f, l in zip(filters, layers)]
        )
        # self.layer1 = self._make_layer(block, filters[0], layers[0])
        # self.layer2 = self._make_layer(block, filters[1], layers[1])
        # self.layer3 = self._make_layer(block, filters[2], layers[2])
        # self.layer4 = self._make_layer(block, filters[3], layers[3])
        # self.layer5 = self._make_layer(block, filters[4], layers[4])
        # self.layer6 = self._make_layer(block, filters[5], layers[5])

        #####
        # aggregate to utterance(segment)-level
        #####
        self.bn_before_agg = nn.BatchNorm1d(filters[-1])
        self.attention = nn.Sequential(
            nn.Conv1d(filters[-1], 128, kernel_size=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, filters[-1], kernel_size=1),
            nn.Softmax(dim=-1),
        )

        #####
        # speaker embedding layer
        #####
        self.fc = nn.Linear(filters[-1] * 2, num_classes)  # speaker embedding layer
        self.lrelu = nn.LeakyReLU(0.3)  # keras style

        #####
        # initialize
        #####
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, nb_layer, downsample_all=False):
        if downsample_all:
            downsamples = [True] * (nb_layer)
        else:
            downsamples = [False] * (nb_layer - 1) + [True]
        layers = []
        for d in downsamples:
            layers.append(block(self.inplanes, planes, downsample=d))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x, is_test=False, label=None):
        # extract speaker embeddings.
        x = self.forward_encoder(x)
        return x

    def forward_encoder(self, x):
        #####
        # first layers before residual blocks
        #####
        x = self.conv1(x)

        #####
        # frame-level
        #####
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.layer5(x)
        # x = self.layer6(x)
        x = self.residual_layers(x)

        #####
        # aggregation: attentive statistical pooling
        #####
        x = self.bn_before_agg(x)
        x = self.lrelu(x)
        w = self.attention(x)
        m = torch.sum(x * w, dim=-1)
        s = torch.sqrt((torch.sum((x**2) * w, dim=-1) - m**2).clamp(min=1e-5))
        x = torch.cat([m, s], dim=1)
        x = x.view(x.size(0), -1)

        #####
        # speaker embedding layer
        #####
        x = self.fc(x)

        return x


def get_RawNet2(layers, filters, num_classes, block=RawNetBasicBlock, **kwargs):
    model = RawNet2(
        RawNetBasicBlock,
        layers=layers,
        filters=filters,
        num_classes=num_classes,
        **kwargs
    )
    return model


class RawNet2Architecture(RawNet2):
    def __init__(self, config, **kwargs):
        self.num_classes = config.num_classes

        if "rawnet_filters" in config:
            filters = config["rawnet_filters"]
        else:
            filters = [128, 128, 256, 256, 256, 256]

        if "rawnet_layers" in config:
            layers = config["rawnet_layers"]
        else:
            layers = [1, 1, 3, 4, 6, 3]

        super().__init__(RawNetBasicBlock, layers, filters, self.num_classes, **kwargs)

        self.fc = nn.Identity()
        self.in_features = filters[-1] * 2

        self.heads = nn.ModuleDict()
        for head in config.heads:
            self.add_head(head)

    def add_head(self, name):
        self.heads.add_module(name, nn.Linear(self.in_features, self.num_classes))

    def forward(self, x):
        x = super().forward(x)
        heads = {name: head(x) for name, head in self.heads.items()}
        return heads


if __name__ == "__main__":
    from torchsummary import summary

    layers = [1, 1, 3, 4, 6, 3]
    layers = [1, 1, 1, 1]
    filters = [64, 128, 256, 512]
    num_classes = 50

    # model = RawNet2Architecture(50)
    model = RawNet2(
        RawNetBasicBlock, layers=layers, filters=filters, num_classes=num_classes
    )

    # summary(model, (1, 11025 * 6), 32,  device='cpu')
    summary(model, (1, 10000 * 5), 64, device="cpu")
