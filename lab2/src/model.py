import torch
import torch.nn as nn


class FeatureCNN(nn.Module):
    # 用于特征提取
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.4, enable_res=True):
        super(FeatureCNN, self).__init__()
        self.enable_res = enable_res

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        if stride != 1 or in_channels != out_channels:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # 此时输入和输出的尺寸是匹配的，因此可以直接进行相加操作
            self.res = nn.Sequential()

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True)
        )

        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        out = self.conv(x)
        # 残差连接
        if self.enable_res:
            out += self.res(x)
        # 因为 self.conv 中第二层卷积层之后可能需要与残差连接，所以这一卷积层的 relu 放在连接之后
        out = self.relu(out)
        out = self.pool(out)
        return out


class Classifier(nn.Module):
    # 用于分类
    def __init__(self, in_features, num_classes=10):
        super(Classifier, self).__init__()
        # 全连接层中一般不使用 BatchNorm，因为BatchNorm会对每个特征进行归一化，可能会去除一些特征之间的关系
        self.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.features = nn.Sequential(
            FeatureCNN(3, 64),
            FeatureCNN(64, 128),
            FeatureCNN(128, 256),
            FeatureCNN(256, 512)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 将最终结果转成 1x1 的矩阵
        self.classifier = Classifier(512, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)  # 将 x 从第一维开始展开，当然可以用 x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
