import os
import torch
from datautils import CIFAR10Data, plot_train_loss, plot_val_data
from model import CIFAR10CNN
from args import args
from train import Trainer, Tester


def train_process():
    model = CIFAR10CNN()

    trainer = Trainer(args, model, train_loader, val_loader)
    all_train_loss, all_val_loss, all_val_acc = trainer.train()

    plot_train_loss(args, all_train_loss)
    plot_val_data(args, all_val_loss, 'Loss')
    plot_val_data(args, all_val_acc, 'Acc')


def test_process():
    model = CIFAR10CNN()
    # 不能直接 torch.load，因为保存的只是一个 state_dict
    if os.path.exists(args.save_path + '/best_model.pkl'):  # 如果存在最佳的模型，则加载最佳的模型
        model.load_state_dict(torch.load(args.save_path + '/best_model.pkl'))
    else:
        model.load_state_dict(torch.load(args.save_path + '/model.pkl'))

    tester = Tester(args, model, test_loader)
    tester.test()


if __name__ == '__main__':
    data = CIFAR10Data(args)
    train_loader, val_loader, test_loader = data.get_data_loader()
    train_process()
    test_process()
