import argparse
import torch
import os
import json

parser = argparse.ArgumentParser(description='deeplearning lab1')

parser.add_argument('--train_batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save_path', type=str, default='../result/', metavar='S',
                    help='path to save the result')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")
args.device = device

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
config_file = open(args.save_path + 'args.json', 'w')
json.dump(args.__dict__, config_file, indent=1)
