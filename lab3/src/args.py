from argparse import ArgumentParser
import torch

parser = ArgumentParser()

parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--patience', type=int, default=20)

args = parser.parse_args()
args.no_cuda = True
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = torch.device('cuda' if args.cuda else 'cpu')
