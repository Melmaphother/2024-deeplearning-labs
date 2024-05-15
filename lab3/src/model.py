import torch
import torch.nn as nn


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias


class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConv(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.gc2 = GraphConv(hidden_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.gc2(x, adj)
        # x = self.bn2(x)
        return x
