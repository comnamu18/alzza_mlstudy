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
from torch.optim import SGD

RND_MEAN = 0
RND_STD = 0.003
RANDOM_SEED = 1234

CONFIG = {
    'batch-size' : 100,
    'epoch' : 50,
    'lr' : 1e-3,
    'report' : 10,
    'num_of_in_features' : 10,
    'num_of_out_features' : 1,
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


def train_and_test(model):
    csv = pd.read_csv("../chap01/abalone.csv")
    train_csv, test_csv = train_test_split(csv, test_size=0.2)

    train_data = AbaloneDataset(train_csv)
    test_data = AbaloneDataset(test_csv)

    train_loader = DataLoader(train_data, batch_size=CONFIG['batch-size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=CONFIG['batch-size'], shuffle=False, drop_last=False)
    
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
    train_and_test(model)