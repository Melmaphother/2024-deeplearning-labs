from model import GCN
from trainer import Trainer, Tester
from datautils import GraphDataLoader
from args import args
import torch


def main():
    data_loader = GraphDataLoader('Cora')
    adj_matrix = data_loader.get_norm_laplacian_matrix()
    model = GCN(in_features=1433, hidden_features=16, out_features=7, dropout=args.dropout)
    trainer = Trainer(args, model, data_loader.get_train_data(), data_loader.get_val_data(), adj_matrix)
    trainer.train()
    # load model
    model.load_state_dict(torch.load('model/best_model.pkl'))
    tester = Tester(model, data_loader.get_test_data(), adj_matrix)
    tester.test()


if __name__ == '__main__':
    main()
