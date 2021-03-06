import sys
sys.path.append('../utils_torch')
from utils import *
from dataloader import *

import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim import SGD

RND_MEAN = 0
RND_STD = 0.003
RANDOM_SEED = 1234

CONFIG = {
    'batch-size' : 100,
    'epoch' : 100,
    'lr' : 1e-2,
    'report' : 20,
    'num_of_in_features' : 10,
    'num_of_out_features' : 1,
}

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(in_features=CONFIG['num_of_in_features'], out_features=CONFIG['num_of_out_features'], bias=True)
        
        with torch.no_grad(): # Grad Tracking Stop
            self.fc1.weight = torch.nn.Parameter(torch.normal(mean=RND_MEAN,std=RND_STD, size=(CONFIG['num_of_out_features'], CONFIG['num_of_in_features'])))
            self.fc1.bias = torch.nn.Parameter(torch.zeros((CONFIG['num_of_out_features'])))
    
    def forward(self, x): 
        return self.fc1(x)


def train_and_test():
    csv = pd.read_csv("./abalone.csv")
    train_csv, test_csv = train_test_split(csv, test_size=0.2)

    train_data = AbaloneDataset(train_csv)
    test_data = AbaloneDataset(test_csv)

    train_loader = DataLoader(train_data, batch_size=CONFIG['batch-size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=CONFIG['batch-size'], shuffle=False, drop_last=False)
    
    model = NeuralNet().cuda()
    optimizer = SGD(model.parameters(), lr=CONFIG['lr'], momentum=0.9)
    criterion = nn.MSELoss()

    for e in range(CONFIG['epoch']):
        train_acc = []
        train_loss = []
        model.train()
        for i, data in enumerate(train_loader):
            x = data['value'].cuda()
            y = data['label'].cuda()
            y = y.view((-1, 1))

            pred = model(x)        
            loss = criterion(pred, y)
            train_loss.append(loss.item())

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            train_acc.append((abs(pred-y)/y).sum().item() / y.size(0))

        train_loss = np.mean(train_loss)
        train_acc = 1 - np.mean(train_acc)

        val_acc = []
        val_loss = []
        model.eval()
        for i, data in enumerate(test_loader):
            x = data['value'].cuda()
            y = data['label'].cuda()
            y = y.view((-1, 1))

            pred = model(x)
            loss = criterion(pred, y)

            val_loss.append(loss.item())
            val_acc.append((abs(pred-y)/y).sum().item() / y.size(0))

        val_loss = np.mean(val_loss)
        val_acc = 1 - np.mean(val_acc)

        if (e+1) % CONFIG['report'] == 0:
            report(e, train_loss, train_acc, val_loss, val_acc)


if __name__ == '__main__':
    set_seed(123)
    train_and_test()


