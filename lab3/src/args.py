from argparse import ArgumentParser
import torch

parser = ArgumentParser()

parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--no_cuda', action='store_true', default=False)

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = torch.device('cuda' if args.cuda else 'cpu')
