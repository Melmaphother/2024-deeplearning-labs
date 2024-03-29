from datautils import CIFAR10Data
from model import CIFAR10CNN
from args import args
from train import Trainer


def main():
    data = CIFAR10Data(args)
    train_loader, val_loader, test_loader = data.get_data_loader()

    model = CIFAR10CNN()

    trainer = Trainer(args, model, train_loader, val_loader, test_loader)
    trainer.train()


if __name__ == '__main__':
    main()
