import torch
from datautils import CIFAR10Data, plot_train_loss, plot_val_data
from model import CIFAR10CNN
from args import args
from train import Trainer, Tester


def main():
    data = CIFAR10Data(args)
    train_loader, val_loader, test_loader = data.get_data_loader()

    model = CIFAR10CNN()

    trainer = Trainer(args, model, train_loader, val_loader)
    all_train_loss, all_val_loss, all_val_acc = trainer.train()
    plot_train_loss(args, all_train_loss)
    plot_val_data(args, all_val_loss, 'Loss')
    plot_val_data(args, all_val_acc, 'Acc')

    # model = torch.load(args.save_path + './model.pkl')
    # tester = Tester(args, model, test_loader)
    # tester.test()


if __name__ == '__main__':
    main()
