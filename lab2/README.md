# 使用卷积神经网络在 CIFAR-10 数据集上实现图片分类

## 关于 CIFAR 数据集

该数据集共有60000张带标签的彩色图像，这些图像尺寸32*32，分为10个类，每类6000张图。

这里面有50000张用于训练，每个类5000张，另外10000用于测试，每个类1000张。

十个类分别为：
```python
cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
```

## 数据处理

1. 使用 ImageNet 预训练权重
2. 数据增强

```python
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

from torchvision import transforms
transform_train = transforms.Compose([transforms.ToTensor(),  # 转为Tensor
                                     transforms.Normalize(norm_mean, norm_std),  # 归一化到[-1,1]
                                     transforms.RandomHorizontalFlip(),  # 随机水平镜像
                                     transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # 随机遮挡
                                     transforms.RandomCrop(32, padding=4)  # 随机中心裁剪
                                    ])

transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(norm_mean, norm_std)])
```

## 训练

- 优化器：SGD，使用动量0.9，权重衰减5e-3
- 学习率动态更新：MultiStepLR，分别在[80, 140]时降低0.1倍
- 记录最佳准确率和对应的 epoch
- 记录所有训练时的 loss 变化和验证时的 loss、acc 变化，并作图

## 模型

- 架构
ResNet18的架构

- 规模

18.6MB

- 使用残差连接网络
避免梯度消失和梯度爆炸，提高网络深度

- 使用全局平均池化
减少参数量，减轻过拟合