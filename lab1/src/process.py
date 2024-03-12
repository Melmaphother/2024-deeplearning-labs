from datautils import DataProcess, load_data, plot_data
from model import FeedForwardNN
from args import args
from train import Trainer
import torch
import torch.utils.data as Data
import numpy as np

func = lambda x: np.log2(x) + np.cos(np.pi * x / 2)

N = 2000

data_process = DataProcess(func, N)
data_process.process()

train_data, val_data, test_data = load_data()
train_data, val_data, test_data = torch.Tensor(train_data), torch.Tensor(val_data), torch.Tensor(test_data)
train_data = Data.TensorDataset(train_data[0], train_data[1])
val_data = Data.TensorDataset(val_data[0], val_data[1])
test_data = Data.TensorDataset(test_data[0], test_data[1])

train_loader = Data.DataLoader(
    train_data, batch_size=args.train_batch_size, shuffle=True)
val_loader = Data.DataLoader(
    val_data, batch_size=args.test_batch_size, shuffle=True)
test_loader = Data.DataLoader(
    test_data, batch_size=args.test_batch_size, shuffle=True)

model = FeedForwardNN(input_size=1, hidden_layers=2, hidden_size=64, output_size=1)

trainer = Trainer(args, model, train_loader, val_loader)
trainer.train()
output = trainer.value()
plot_data(output=output, func=func, N=N)