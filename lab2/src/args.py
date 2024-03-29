import argparse
import torch
import os
import json
from pathlib import Path

parser = argparse.ArgumentParser(description='deeplearning lab2')

# dataset args
parser.add_argument('--train_batch_size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test_batch_size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--val_batch_size', type=int, default=256, metavar='N',
                    help='input batch size for validation (default: 256)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')

# train and val args
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=5e-3, metavar='W',
                    help='SGD weight decay (default: 5e-3)')
parser.add_argument('--step_size', type=int, default=5, metavar='S',
                    help='step size for lr decay (default: 5)')
parser.add_argument('--gamma', type=float, default=0.1, metavar='G',
                    help='gamma for lr decay (default: 0.1)')
parser.add_argument('--val_interval', type=int, default=1, metavar='V',
                    help='interval of validation (default: 1)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training (default: False)')

# data path args
parser.add_argument('--root_path', type=str, default='../data/', metavar='R',
                    help='path of CIFAR-10 dataset')
parser.add_argument('--save_path', type=str, default='../result/2/', metavar='S',
                    help='path to save the result')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

args.root_path = str(Path(args.root_path).resolve())
args.save_path = str(Path(args.save_path).resolve())

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
config_file = open(args.save_path + '/args.json', 'w')
json.dump(args.__dict__, config_file, indent=1)

device = torch.device("cuda" if args.cuda else "cpu")
args.device = device
