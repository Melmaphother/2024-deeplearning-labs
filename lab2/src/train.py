import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class Trainer:
    def __init__(self, args, model, train_loader, val_loader):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(args.device)

        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum,
                                   weight_decay=args.weight_decay)  # weight_decay: L2正则
        self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                        milestones=[int(args.epochs * 0.4), int(args.epochs * 0.7)],
                                                        gamma=args.gamma, last_epoch=-1)
        self.best_acc = 0
        self.best_epoch = 0

    def _train_single_epoch(self):
        self.model.train()
        train_loss = 0
        tqdm_train_loader = tqdm(self.train_loader)

        for batch_idx, (inputs, labels) in enumerate(tqdm_train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(self.train_loader)
        print(f'Train loss: {avg_train_loss: .4f}')
        return avg_train_loss

    def _validate(self):
        self.model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        tqdm_val_loader = tqdm(self.val_loader)

        with torch.no_grad():
            for inputs, labels in tqdm_val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicts = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicts == labels).sum().item()

        avg_val_loss = val_loss / len(self.val_loader)
        accuracy = 100 * val_correct / val_total
        print(f'Validation loss: {avg_val_loss: .4f}, accuracy: {accuracy: .2f}%')

        return avg_val_loss, accuracy

    def train(self):
        all_train_loss = []
        all_val_loss = []
        all_val_acc = []
        with open(self.args.save_path + '/val_info.txt', 'w'):
            pass

        for epoch in range(self.args.epochs):
            print(f'Epoch {epoch + 1}/{self.args.epochs}')
            train_loss = self._train_single_epoch()
            all_train_loss.append(train_loss)

            if (epoch + 1) % self.args.val_interval == 0:
                val_loss, accuracy = self._validate()
                all_val_loss.append(val_loss)
                all_val_acc.append(accuracy)

                if self.best_acc < accuracy:
                    self.best_acc = accuracy
                    self.best_epoch = epoch

                with open(self.args.save_path + '/val_info.txt', 'a') as f:
                    f.write(f'epoch: {epoch + 1}, val_loss: {val_loss: .4f}, val_acc: {accuracy: .2f}%\n')

            self.scheduler.step()
            print('last_lr:', self.scheduler.get_last_lr())

        print(f'Best val acc: {self.best_acc: .2f}%, epoch: {self.best_epoch + 1}')
        with open(self.args.save_path + '/val_info.txt', 'a') as f:
            f.write(f'Best val acc: {self.best_acc: .2f}%, epoch: {self.best_epoch + 1}\n')
        torch.save(self.model.state_dict(), self.args.save_path + '/model.pkl')
        return all_train_loss, all_val_loss, all_val_acc


class Tester:
    def __init__(self, args, model, test_loader):
        self.args = args
        self.model = model
        self.test_loader = test_loader
        self.device = torch.device(args.device)

        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

    def test(self):
        test_loss = 0
        test_correct = 0
        test_total = 0
        tqdm_test_loader = tqdm(self.test_loader)

        with torch.no_grad():
            for inputs, labels in tqdm_test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                test_loss += loss.item()
                _, predicts = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicts == labels).sum().item()

        test_loss /= len(self.test_loader)
        accuracy = 100 * test_correct / test_total

        print(f'Test loss: {test_loss: .4f}, accuracy: {accuracy: .2f}%')
