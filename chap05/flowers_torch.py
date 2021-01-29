import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from torch.optim import SGD
import torch.nn.functional as F
from torchvision import transforms
import random

import sys
sys.path.append('../utils_torch')
from utils import *
from dataloader import *
from network import *
from config import *

def train_and_test(model, train_loader, validation_loader, args):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for e in range(args.epoch):
        train_metric = []
        train_loss = []
        model.train()
        for i, data in enumerate(train_loader):
            x = data['x'].cuda()
            y = data['y'].cuda()

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
        for i, data in enumerate(validation_loader):
            x = data['x'].cuda()
            y = data['y'].cuda()

            pred = model(x)

            loss = criterion(pred, y)
            val_loss.append(loss.item())
            
            # 1. 직접계산 
            output = pred.detach()
            output = torch.argmax(output, dim=1)

            # print(output)
            # print(y)

            acc = torch.where(output == y, 1, 0).sum().item() / y.size(0)
            val_metric.append(acc)

        val_loss = np.mean(val_loss)
        val_metric = np.mean(val_metric, axis=0)

        if (e+1) % args.report == 0:
            report(e, train_loss, train_metric, val_loss, val_metric)


if __name__ == '__main__':
    args = flower_mlp_config()

    fdata = FlowerDataset()
    n_fdata = len(fdata)
    split = int(n_fdata * 0.8)
    indices = list(range(n_fdata))
    random.shuffle(indices)

    train_sampler = SubsetRandomSampler(indices[:split])
    validation_sampler = SubsetRandomSampler(indices[split:])
    
    train_loader = DataLoader(fdata, batch_size=args.batchsize, sampler=train_sampler)
    validation_loader = DataLoader(fdata, batch_size=args.batchsize, sampler=validation_sampler)

    set_seed(args.seed)
    model = mlp(args).cuda()
    summary(model, input_size=(args.in_features,))

    train_and_test(model, train_loader, validation_loader, args)



