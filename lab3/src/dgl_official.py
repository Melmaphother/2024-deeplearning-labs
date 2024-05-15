import argparse

import dgl
import dgl.nn.pytorch as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from model import GCN
from dgl.nn.pytorch import GraphConv


# class GCN(nn.Module):
#     def __init__(self, in_size, hid_size, out_size):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         # two-layer GCN
#         self.layers.append(
#             dglnn.GraphConv(in_size, hid_size, activation=F.relu)
#         )
#         self.layers.append(dglnn.GraphConv(hid_size, out_size))
#         self.dropout = nn.Dropout(0.5)
# 
#     def forward(self, g, features):
#         h = features
#         for i, layer in enumerate(self.layers):
#             if i != 0:
#                 h = self.dropout(h)
#             h = layer(g, h)
#         return h


def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(features, g)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g, features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training loop
    for epoch in range(200):
        model.train()
        logits = model(features, g)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(g, features, labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )


def get_norm_laplacian_matrix(g):
    _A = g.adjacency_matrix().to_dense().numpy().astype(np.float32)
    _I = np.eye(_A.shape[0])
    _D = np.diag(np.power(_A.sum(1), -0.5).flatten(), 0)
    _L = _I - _D @ _A @ _D
    return torch.FloatTensor(_A)


import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'citeseer', 'pubmed').",
    )
    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type(float, bfloat16)",
    )
    args = parser.parse_args()
    print(f"Training with DGL built-in GraphConv module.")

    # load and preprocess dataset
    transform = (
        AddSelfLoop()
    )  # by default, it will first remove self-loops to prevent duplication
    if args.dataset == "cora":
        data = CoraGraphDataset()
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset()
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset()
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    g = data[0]
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    g = g.int().to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]
    adj = get_norm_laplacian_matrix(g)
    adj = torch.FloatTensor(adj).to(device)

    # create GCN model
    in_size = features.shape[1]
    out_size = data.num_classes
    model = GCN(in_size, 16, out_size, 0.5).to(device)

    # convert model and graph to bfloat16 if needed
    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        features = features.to(dtype=torch.bfloat16)
        model = model.to(dtype=torch.bfloat16)

    # model training
    print("Training...")
    train(adj, features, labels, masks, model)

    # test the model
    print("Testing...")
    acc = evaluate(adj, features, labels, masks[2], model)
    print("Test accuracy {:.4f}".format(acc))
