import sys
sys.path.append('../utils_torch')
from utils import *
from dataloader import *

import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from torch.optim import SGD
import torch.nn.functional as F

RND_MEAN = 0
RND_STD = 0.003
RANDOM_SEED = 1234

CONFIG = {
    'batch-size' : 1,
    'epoch' : 10,
    'lr' : 1e-3,
    'report' : 1,
    'num_of_in_features' : 27,
    'num_of_out_features' : 7,
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
    csv = pd.read_csv("./faults.csv")
    csv = extend_data(csv)
    train_csv, test_csv = train_test_split(csv, test_size=0.2)

    train_data = SteelPlateDataset(train_csv)
    test_data = SteelPlateDataset(test_csv)

    train_loader = DataLoader(train_data, batch_size=CONFIG['batch-size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, drop_last=False)
    
    model = NeuralNet().cuda()
    optimizer = SGD(model.parameters(), lr=CONFIG['lr'], momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()

    for e in range(CONFIG['epoch']):
        train_metric = []
        train_loss = []
        model.train()
        for i, data in enumerate(train_loader):
            x = data['value'].cuda()
            y = data['label'].cuda()
            # print(x.shape)
            # print(y)
            y = torch.argmax(y, dim=1)
            # print(y)
            # quit()

            pred = model(x)
            # print(pred)
            # quit()             
            loss = criterion(pred, y)
            # print(loss)
            # quit()
            train_loss.append(loss.item())

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            # 1. 직접계산 
            output = pred.detach()
            output = torch.argmax(output, dim=1)
            acc = torch.where(output == y, 1, 0).sum().item() / y.size(0)
            train_metric.append(acc)

        train_loss = np.mean(train_loss)
        train_metric = np.mean(train_metric, axis=0)

        val_metric = []
        val_loss = []
        model.eval()
        for i, data in enumerate(test_loader):
            x = data['value'].cuda()
            y = data['label'].cuda()
            y = torch.argmax(y, dim=1)

            pred = model(x)
            loss = criterion(pred, y)
            val_loss.append(loss.item())
            
            # 1. 직접계산 
            output = pred.detach()
            output = torch.argmax(output, dim=1)
            acc = torch.where(output == y, 1, 0).sum().item() / y.size(0)
            val_metric.append(acc)

        val_loss = np.mean(val_loss)
        val_metric = np.mean(val_metric, axis=0)

        if (e+1) % CONFIG['report'] == 0:
            report(e, train_loss, train_metric, val_loss, val_metric)


if __name__ == '__main__':
    set_seed(123)
    train_and_test()
