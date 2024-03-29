import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# 加载完整的训练数据集
train_data = datasets.CIFAR10(root='./data', train=True, download=False, transform=transforms.ToTensor())
test_data = datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())

# 定义训练集和验证集的大小比例
train_size = int(0.8 * len(train_data))  # 假设使用80%的数据作为训练集
val_size = len(train_data) - train_size  # 剩下的20%作为验证集

# 划分训练集和验证集
train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

print(f"Training data samples: {len(train_loader.dataset)}")
print(f"Validation data samples: {len(val_loader.dataset)}")
