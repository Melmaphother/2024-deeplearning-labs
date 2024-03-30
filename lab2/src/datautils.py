import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np


class CIFAR10Data:
    def __init__(self, args):
        self.seed = args.seed
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.val_batch_size = args.val_batch_size

        # Imagenet数据集的均值和标准差（预训练权重）
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([transforms.ToTensor(),  # 转为Tensor
                                              transforms.Normalize(norm_mean, norm_std),  # 归一化到[-1,1]
                                              transforms.RandomHorizontalFlip(),  # 随机水平镜像
                                              transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # 随机遮挡
                                              transforms.RandomCrop(32, padding=4)  # 随机中心裁剪
                                              ])

        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(norm_mean, norm_std)])

        self.train_data = datasets.CIFAR10(root=args.root_path, train=True, download=True,
                                           transform=transform_train)
        self.test_data = datasets.CIFAR10(root=args.root_path, train=False, download=True,
                                          transform=transform_test)

        # 划分训练集和验证集
        self.train_dataset, self.val_dataset = self._split_train_val(self.train_data, train_ratio=0.8)

    def _split_train_val(self, dataset, train_ratio=0.8):
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(self.seed))

    def get_data_loader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False)
        test_loader = DataLoader(self.test_data, batch_size=self.test_batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


def plot_val_data(args, data, description='Loss'):
    epochs = np.arange(args.val_interval, args.epochs + 1, args.val_interval)
    data = np.array(data)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, data, label='Validation ' + description)
    plt.title('Validation ' + description)
    plt.xlabel('Epoch')
    plt.ylabel(description)
    plt.legend()
    plt.savefig(args.save_path + '/validation_' + description.lower() + '.png')
    plt.show()


def plot_train_loss(args, data):
    epochs = np.arange(1, args.epochs + 1)
    data = np.array(data)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, data, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(args.save_path + '/training_loss.png')
    plt.show()
