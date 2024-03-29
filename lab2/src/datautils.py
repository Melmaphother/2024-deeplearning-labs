import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np


class CIFAR10Data:
    def __init__(self, args):
        self.root_dir = args.root_dir
        self.seed = args.seed
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.val_batch_size = args.val_batch_size

        self.train_data = datasets.CIFAR10(root=self.root_dir, train=True, download=False,
                                           transform=transforms.ToTensor())
        self.test_data = datasets.CIFAR10(root=self.root_dir, train=False, download=False,
                                          transform=transforms.ToTensor())

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


def plot_loss(args, all_loss):
    """
    绘制训练过程中的损失变化图。

    Parameters:
    - args: 包含训练参数的对象，应该有epochs和val_interval属性。
    - all_loss: 包含每次验证损失的列表。
    """
    # 根据验证间隔和总的epochs数计算x轴的坐标点
    epochs = np.arange(args.val_interval, args.epochs + 1, args.val_interval)
    all_loss = np.array(all_loss)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, all_loss, label='Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
