import sys
sys.path.append('../utils_torch')
from utils import *
from dataloader import *
import argparse
from torchsummary import summary

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
    'batch-size' : 128,
    'epoch' : 50,
    'lr' : 1e-3,
    'report' : 10,
    'num_of_in_features' : 27,
    'num_of_out_features' : 7,
}

class NeuralNet(nn.Module):
    def __init__(self, args):
        super(NeuralNet, self).__init__()
        modules = []
        prev = CONFIG['num_of_in_features']        
        for h in args.hidden:
            modules.append(nn.Linear(in_features=prev, out_features=h, bias=True))
            modules.append(nn.ReLU())
            prev = h
        modules.append(nn.Linear(in_features=prev, out_features=CONFIG['num_of_out_features'], bias=True))

        self.net = nn.Sequential(*modules)
        
        # initialize weight and bias
        for layer in self.net:
            if type(layer) is nn.Linear:
                nn.init.normal_(layer.weight, mean=RND_MEAN, std=RND_STD)
                nn.init.normal_(layer.bias, mean=RND_MEAN, std=RND_STD)
    
    def forward(self, x):
        return self.net(x)


def train_and_test(model, args):
    csv = pd.read_csv("../chap03/faults.csv")
    csv = extend_data(csv)
    train_csv, test_csv = train_test_split(csv, test_size=0.2)

    train_data = SteelPlateDataset(train_csv)
    test_data = SteelPlateDataset(test_csv)

    train_loader = DataLoader(train_data, batch_size=CONFIG['batch-size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, drop_last=False)
    
    optimizer = SGD(model.parameters(), lr=CONFIG['lr'], momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for e in range(CONFIG['epoch']):
        train_metric = []
        train_loss = []
        model.train()
        for i, data in enumerate(train_loader):
            x = data['value'].cuda()
            y = data['label'].cuda()
            y = torch.argmax(y, dim=1)

            pred = model(x)
            loss = criterion(pred, y)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=str, default=None)
    args = parser.parse_args()

    if args.hidden is None:
        args.hidden = []
    else:
        args.hidden = [int(h.strip()) for h in args.hidden.split(',')]

    set_seed(123)
    model = NeuralNet(args).cuda()
    summary(model, input_size=(CONFIG['num_of_in_features'],))
    train_and_test(model, args)
