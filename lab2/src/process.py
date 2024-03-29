from datautils import CIFAR10Data, plot_loss
from model import CIFAR10CNN
from args import args
from train import Trainer, Tester
import torch


def main():
    data = CIFAR10Data(args)
    train_loader, val_loader, test_loader = data.get_data_loader()

    model = CIFAR10CNN()

    trainer = Trainer(args, model, train_loader, val_loader)
    all_val_loss = trainer.train()
    plot_loss(args, all_val_loss)

    # model = torch.load(args.save_path + './model.pkl')
    # tester = Tester(args, model, test_loader)
    # tester.test()


if __name__ == '__main__':
    main()
