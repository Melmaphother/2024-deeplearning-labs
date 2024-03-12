import torch
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt

class DataProcess():
    def __init__(self, func, N, range_x=(1, 16), train_radio=0.8, val_radio=0.1, test_radio=0.1, random_seed=42):
        self.func = func
        self.N = N
        self.range_x = range_x
        self.train_radio = train_radio
        self.val_radio = val_radio
        self.test_radio = test_radio
        self.random_seed = random_seed
        self.X = torch.linspace(self.range_x[0], self.range_x[1], self.N).contiguous().view(-1, 1)
        self.train_data = {}
        self.val_data = {}
        self.test_data = {}

    def split_samples(self):
        # 注意这里需要强制转化成 int，否则 random_split 会报错
        val_size = int(self.N * self.val_radio)
        test_size = int(self.N * self.test_radio)
        train_size = self.N - val_size - test_size
        train_samples, val_samples, test_samples = random_split(self.X, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(self.random_seed))

        return train_samples, val_samples, test_samples

    def generate_labels(self, samples):
        train_samples, val_samples, test_samples = samples
        train_labels = self.func(torch.Tensor(train_samples))
        val_labels = self.func(torch.Tensor(val_samples))
        test_labels = self.func(torch.Tensor(test_samples))
        return train_labels, val_labels, test_labels
    
    def save_data(self, samples, labels, path='./lab1/data/'):
        train_samples, val_samples, test_samples = samples
        train_labels, val_labels, test_labels = labels
        self.train_data['samples'] = train_samples
        self.train_data['labels'] = train_labels
        self.val_data['samples'] = val_samples
        self.val_data['labels'] = val_labels
        self.test_data['samples'] = test_samples
        self.test_data['labels'] = test_labels
        torch.save(self.train_data, path + 'train_data.pt')
        torch.save(self.val_data, path + 'val_data.pt')
        torch.save(self.test_data, path + 'test_data.pt')
    
    def process(self):
        samples = self.split_samples()
        labels = self.generate_labels(samples)
        self.save_data(samples, labels)


def load_data():
        train_data = torch.load('./lab1/data/train_data.pt')
        val_data = torch.load('./lab1/data/val_data.pt')
        test_data = torch.load('./lab1/data/test_data.pt')
        return [train_data['samples'], train_data['labels']], [val_data['samples'], val_data['labels']], [test_data['samples'], test_data['labels']]

def plot_data(output, func, N, range_x=(1, 16)):
    x = np.linspace(range_x[0], range_x[1], 10 * N)
    y = func(x)
    all_input, all_target, all_output = output
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='log2(x) + cos(pi*x/2)', color='black')  # 函数图像
    plt.scatter(all_input, all_target, label='Targets', color='green', alpha=0.6)  # 目标值散点图
    plt.scatter(all_input, all_output, label='Outputs', color='red', alpha=0.6)  # 模型输出散点图
    plt.title('Function Visualization with Targets and Outputs')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig('./lab1/result/visualization.png')
    plt.show()