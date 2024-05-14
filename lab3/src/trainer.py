import torch
from sklearn.metrics import accuracy_score
import os


class Trainer:
    def __init__(self, args, model, train_loader, val_loader, adj_matrix):
        self.args = args
        self.device = args.device
        self.num_epochs = args.num_epochs
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.adj_matrix = adj_matrix
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        best_val_acc = 0
        for epoch in range(self.num_epochs):
            self.model.train()
            features, labels = self.train_loader
            features, labels = features.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(features, self.adj_matrix)
            train_loss = self.criterion(output, labels)
            train_acc = accuracy_score(labels.cpu().numpy(), output.argmax(dim=1).cpu().numpy())
            train_loss.backward()
            self.optimizer.step()
            print(f'Epoch: {epoch + 1:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

            val_loss, val_acc = self.val()
            print(f'Epoch: {epoch + 1:03d}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if not os.path.exists('model'):
                    os.makedirs('model')
                torch.save(self.model.state_dict(), 'model/best_model.pkl')

    def val(self):
        self.model.eval()
        with torch.no_grad():
            features, labels = self.val_loader
            features, labels = features.to(self.device), labels.to(self.device)
            output = self.model(features, self.adj_matrix)
            loss = self.criterion(labels.cpu().numpy(), output.argmax(dim=1).cpu().numpy())
            acc = accuracy_score(labels.numpy(), output.argmax(dim=1).numpy())
        return loss, acc


class Tester:
    def __init__(self, args, model, test_loader, adj_matrix):
        self.model = model.to(args.device)
        self.device = args.device
        self.test_loader = test_loader
        self.adj_matrix = adj_matrix

    def test(self):
        self.model.eval()
        with torch.no_grad():
            features, labels = self.test_loader
            features, labels = features.to(self.device), labels.to(self.device)
            output = self.model(features, self.adj_matrix)
            acc = accuracy_score(labels.cpu().numpy(), output.argmax(dim=1).cpu().numpy())
        print(f'Test Acc: {acc:.4f}')
