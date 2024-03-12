import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, hidden_layers=2, activation=F.relu()):
        super(FeedForwardNN, self).__init__()
        self.activation = activation
        # cat input layer and the first hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        # cat hidden layers
        for i in range(hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        # cat the last hidden layer and the output layer
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output(x)
        return x