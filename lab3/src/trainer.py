import torch
import os
from sklearn.metrics import accuracy_score


class Trainer:
    def __init__(self, args, model, features, labels, adj_matrix, train_mask, val_mask):
        self.args = args
        self.device = args.device
        self.num_epochs = args.num_epochs
        self.patience = args.patience
        self.model = model.to(self.device)
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)
        self.adj_matrix = adj_matrix.to(self.device)
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        best_val_acc = 0
        counter = 0
        for epoch in range(self.num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(self.features, self.adj_matrix)
            train_loss = self.criterion(output[self.train_mask], self.labels[self.train_mask])
            train_acc = accuracy_score(self.labels[self.train_mask].cpu().numpy(),
                                       output[self.train_mask].argmax(dim=1).cpu().numpy())
            train_loss.backward()
            self.optimizer.step()
            print(f'Epoch: {epoch + 1:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

            val_loss, val_acc = self.val()
            print(f'Epoch: {epoch + 1:03d}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                counter = 0
                if not os.path.exists('../model'):
                    os.makedirs('../model')
                torch.save(self.model.state_dict(), '../model/best_model.pkl')
            else:
                counter = 1

            if counter == self.patience:
                print(f'Early stopping at epoch: {epoch + 1}')
                break

    def val(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.features, self.adj_matrix)
            loss = self.criterion(output[self.val_mask], self.labels[self.val_mask])
            acc = accuracy_score(self.labels[self.val_mask].cpu().numpy(),
                                 output[self.val_mask].argmax(dim=1).cpu().numpy())
        return loss, acc


class Tester:
    def __init__(self, args, model, features, labels, adj_matrix, test_mask):
        self.model = model.to(args.device)
        self.device = args.device
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)
        self.adj_matrix = adj_matrix.to(self.device)
        self.test_mask = test_mask

    def test(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.features, self.adj_matrix)
            acc = accuracy_score(self.labels[self.test_mask].cpu().numpy(),
                                 output[self.test_mask].argmax(dim=1).cpu().numpy())
        print(f'Test Acc: {acc:.4f}')
