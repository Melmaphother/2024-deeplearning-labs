from datautils import DataProcess, load_data
from model import FeedForwardNN
from args import args
from train import Trainer
import torch
import torch.utils.data as Data


def func(x): return torch.log2(x) + torch.cos(torch.pi * x / 2)

N = 200

data_process = DataProcess(func, N)
data_process.process()

train_data, val_data, test_data = load_data()
train_data = Data.TensorDataset(train_data['samples'], train_data['labels'])
val_data = Data.TensorDataset(val_data['samples'], val_data['labels'])
test_data = Data.TensorDataset(test_data['samples'], test_data['labels'])

train_loader = Data.DataLoader(
    train_data, batch_size=args.train_batch_size, shuffle=True)
val_loader = Data.DataLoader(
    val_data, batch_size=args.test_batch_size, shuffle=True)
test_loader = Data.DataLoader(
    test_data, batch_size=args.test_batch_size, shuffle=True)

model = FeedForwardNN(input_size=1, hidden_layers=2, hidden_size=64, output_size=1)

trainer = Trainer(args, model, train_loader, val_loader)
trainer.train()
trainer.value()
