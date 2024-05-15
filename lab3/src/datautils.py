from dgl.data import (
    CoraGraphDataset,
    CiteseerGraphDataset,
)

import numpy as np
import torch
from dgl import AddSelfLoop


class GraphDataLoader:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.__load_data()

    def __load_data(self):
        transform = (
            AddSelfLoop()
        )
        if self.dataset_name == 'Cora':
            dataset = CoraGraphDataset(transform=transform)
        elif self.dataset_name == 'Citeseer':
            dataset = CiteseerGraphDataset(transform=transform)
        else:
            raise ValueError('Unknown dataset: {}'.format(self.dataset_name))

        self.graph = dataset[0]
        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']
        self.train_mask = self.graph.ndata['train_mask']
        self.val_mask = self.graph.ndata['val_mask']
        self.test_mask = self.graph.ndata['test_mask']

    def get_norm_laplacian_matrix(self):
        _A = self.graph.adjacency_matrix().to_dense().numpy().astype(np.float32)
        _I = np.eye(_A.shape[0])
        _D = np.diag(np.power(_A.sum(1), -0.5).flatten(), 0)
        _L = _I + _D @ _A @ _D
        return torch.FloatTensor(_L)

    def get_all_data(self):
        return self.features, self.labels

    def get_train_mask(self):
        return self.train_mask

    def get_val_mask(self):
        return self.val_mask

    def get_test_mask(self):
        return self.test_mask
