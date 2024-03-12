import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np


class Trainer():
    def __init__(self, args, model, train_dataloader, val_dataloader):
        self.args = args
        self.model = model.to(torch.device(self.args.device))
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.epochs = self.args.epochs
        self.lr = self.args.lr
        self.criteria = nn.MSELoss()
        self.optimize = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        print('train')
        for epoch in range(self.epochs):
            self.model.train()
            tqdm_dataloader = tqdm(self.train_dataloader)
            loss_sum = 0
            for idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.args.device) for x in batch]
                input, target = batch
                input, target = input.view(-1, 1), target.view(-1, 1)
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
        all_input = []
        all_target = []
        all_output = []
        for idx, batch in enumerate(self.val_dataloader):
            batch = [x.to(self.args.device) for x in batch]
            input, target = batch
            all_input.extend(input.cpu().detach().numpy())
            all_target.extend(target.cpu().detach().numpy())
            output = self.model(input.view(-1, 1))
            all_output.extend(output.cpu().detach().numpy())
            loss = self.criteria(output, target.view(-1, 1))
            loss_sum += loss.item()

        average_val_loss = loss_sum / len(self.val_dataloader)
        print('val loss: {}'.format(average_val_loss))

        all_input, all_target, all_output = np.array(all_input), np.array(all_target), np.array(all_output)
        difference = all_output - all_target
        with open(self.args.save_path + 'val_result.txt', 'w') as file:
            file.write(f'Average Validation Loss: {average_val_loss}\n')
            for target, output, difference in zip(all_target, all_output, difference):
                file.write(f"{target} {output} {difference}\n")
        
        return all_input, all_target, all_output


