import torch
from torch.utils.data import random_split

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
        val_size = self.N * self.val_radio
        test_size = self.N * self.test_radio
        train_size = self.N - val_size - test_size

        train_samples, val_samples, test_samples = random_split(self.X, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(self.random_seed))

        return train_samples, val_samples, test_samples
    
    def generate_labels(self, samples):
        train_samples, val_samples, test_samples = samples
        train_labels = self.func(train_samples)
        val_labels = self.func(val_samples)
        test_labels = self.func(test_samples)
        return train_labels, val_labels, test_labels
    
    def save_data(self, samples, labels, path='../data/'):
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


def load_data(self):
        train_data = torch.load('../data/train_data.pt')
        val_data = torch.load('../data/val_data.pt')
        test_data = torch.load('../data/test_data.pt')
        return [train_data['samples'], train_data['labels']], [val_data['samples'], val_data['labels']], [test_data['samples'], test_data['labels']]