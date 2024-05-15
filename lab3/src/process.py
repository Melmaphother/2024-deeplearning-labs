from model import GCN
from trainer import Trainer, Tester
from datautils import GraphDataLoader
from args import args
import torch

configs = {
    'Cora': {
        'in_features': 1433,
        'hidden_features': 16,
        'out_features': 7,
        'dropout': args.dropout
    },
    'Citeseer': {
        'in_features': 3703,
        'hidden_features': 16,
        'out_features': 6,
        'dropout': args.dropout
    }
}


def main(dataset_name: str = 'Cora'):
    data_loader = GraphDataLoader(dataset_name)
    assert dataset_name in configs, f'No configuration for dataset: {dataset_name}'
    adj_matrix = data_loader.get_norm_laplacian_matrix()
    features, labels = data_loader.get_all_data()
    model = GCN(**configs[dataset_name])
    trainer = Trainer(args, model, features, labels, adj_matrix, data_loader.get_train_mask(),
                      data_loader.get_val_mask())
    trainer.train()
    # load model
    model.load_state_dict(torch.load('../model/best_model.pkl'))
    tester = Tester(args, model, features, labels, adj_matrix, data_loader.get_test_mask())
    tester.test()


if __name__ == '__main__':
    main('Cora')
