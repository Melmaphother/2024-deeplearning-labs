import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt


class Trainer():
    def __init__(self, args, model, train_dataloader, val_dataloader):
        self.args = args
        self.model = model.to(torch.device(self.args.device))
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.num_epochs = self.args.num_epochs
        self.lr = self.args.lr
        self.criteria = nn.MSELoss()
        self.optimize = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        print('train')
        for epoch in range(self.num_epochs):
            self.model.train()
            tqdm_dataloader = tqdm(self.train_dataloader)
            loss_sum = 0
            for idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.args.device) for x in batch]
                input, target = batch
                input, target = input.to(torch.device(self.args.device)), target.to(
                    torch.device(self.args.device))
                self.optimize.zero_grad()
                output = self.model(input)
                loss = self.criteria(output, target)
                loss.backward()
                self.optimize.step()
                loss_sum += loss.item()
                tqdm_dataloader.set_postfix(loss=loss.item())
                print('epoch: {}, loss: {}'.format(
                    epoch + 1, loss_sum / len(tqdm_dataloader)))

        torch.save(self.model.state_dict(), self.args.save_path + 'model.pkl')

    def value(self):
        print('value')
        self.model.eval()
        loss_sum = 0
        all_target = []
        all_output = []
        for idx, batch in enumerate(self.val_dataloader):
            input, target = batch
            all_target.extend(target.numpy())
            input, target = input.to(torch.device(self.args.device)), target.to(
                torch.device(self.args.device))
            output = self.model(input)
            all_output.extend(output.numpy())
            loss = self.criteria(output, target)
            loss_sum += loss.item()

        average_val_loss = loss_sum / len(self.val_dataloader)
        print('val loss: {}'.format(average_val_loss))

        with open(self.args.save_path + 'val_result.txt', 'w') as file:
            file.write(f'Average Validation Loss: {average_val_loss}\n')

        plt.figure(figsize=(10, 6))
        plt.scatter(all_target, all_output, color='r', alpha=0.5, label='Predictions vs. Actual')
        plt.plot([0, 18], [0, 18], 'b--', label='Ideal')
        plt.xlim(0, 18)
        plt.ylim(-1, 6)
        plt.xlabel('Actual Values')
        plt.ylabel('Predictions')
        plt.title('Validation Set Predictions vs. Actual Values')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.args.save_path + 'plot.png')
        plt.show()