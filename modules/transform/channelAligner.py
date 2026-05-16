from modules.layers.conv import conv, conv1x1, conv3x3, deconv
from torch import nn


class Channel_aligner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            conv3x3(64, 256),
            nn.LeakyReLU(inplace=True),
            conv3x3(256, 256),
            nn.LeakyReLU(inplace=True),
            conv3x3(256, 256),
            nn.LeakyReLU(inplace=True),
            conv3x3(256, 256),
            nn.LeakyReLU(inplace=True),
        )

        self.conv2 = conv3x3(256, 64)
        self.conv3 = conv3x3(256, 64)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)  
        self.avgpool2 = nn.AdaptiveAvgPool2d(1)

    # feature2 is guided
    def forward(self, feature1, feature2):
        identity = feature2

        # 计算beta
        out1 = self.conv1(feature1)
        out1 = self.conv2(out1)
        beta = self.avgpool1(out1)

        # 计算gamma
        out2 = self.conv1(feature2)
        out2 = self.conv3(out2)
        gamma = self.avgpool2(out2)

        # 池化to广播
        out = gamma * identity + beta  
        # print("beta,gamma:")
        # print(beta,gamma)
        return out, beta, gamma  
