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

RND_MEAN = 0
RND_STD = 0.003
RANDOM_SEED = 1234

CONFIG = {
    'batch-size' : 128,
    'epoch' : 10,
    'lr' : 1e-4,
    'report' : 1,
    'num_of_in_features' : 8,
    'num_of_out_features' : 1,
}

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(in_features=CONFIG['num_of_in_features'], out_features=CONFIG['num_of_out_features'], bias=True)
        # self.sigmoid = nn.Sigmoid()

        with torch.no_grad(): # Grad Tracking Stop
            self.fc1.weight = torch.nn.Parameter(torch.normal(mean=RND_MEAN,std=RND_STD, size=(1, 8)))
            self.fc1.bias = torch.nn.Parameter(torch.zeros((1)))
    
    def forward(self, x):
        # retrun self.sigmid(self.fc1(x))
        return self.fc1(x)


def train_and_test():
    csv = pd.read_csv("./pulsar_stars.csv")
    csv = extend_data(csv)
    train_csv, test_csv = train_test_split(csv, test_size=0.2)

    train_data = PulsarDataset(train_csv)
    test_data = PulsarDataset(test_csv)

    train_loader = DataLoader(train_data, batch_size=CONFIG['batch-size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, drop_last=False)
    
    model = NeuralNet().cuda()
    optimizer = SGD(model.parameters(), lr=CONFIG['lr'], momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()

    for e in range(CONFIG['epoch']):
        train_metric = []
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

            # 1. 직접계산 
            output = pred.detach().cpu()
            output = np.where(output >= 0.5, 1, 0).reshape(-1)
            target = y.detach().cpu().numpy().reshape(-1)
            train_metric.append(eval_acc(output, target))

        train_loss = np.mean(train_loss)
        train_metric = np.mean(train_metric, axis=0)

        val_metric = []
        val_loss = []
        model.eval()
        for i, data in enumerate(test_loader):
            x = data['value'].cuda()
            y = data['label'].cuda()
            y = y.view((-1, 1))

            pred = model(x)
            loss = criterion(pred, y)
            val_loss.append(loss.item())
            
            # 1. 직접계산 
            output = pred.detach().cpu()
            output = np.where(output >= 0.5, 1, 0).reshape(-1)
            target = y.detach().cpu().numpy().reshape(-1)
            val_metric.append(eval_acc(output, target))
            
            # 2. scikit-learn 사용 1
            print(accuracy_score(target, output))
            print(precision_score(target, output))
            print(recall_score(target, output))
            print(f1_score(target, output))

            # 3. scikit-learn 사용 2
            print(precision_recall_fscore_support(target, output))

        val_loss = np.mean(val_loss)
        val_metric = np.mean(val_metric, axis=0)

        if (e+1) % CONFIG['report'] == 0:
            report(e, train_loss, train_metric, val_loss, val_metric)


def eval_acc(pred, gt):

    est_true = np.greater(pred, 0)
    est_false = np.logical_not(est_true)
    gt_true = np.greater(gt, 0)
    gt_false = np.logical_not(gt_true)

    TP = np.sum(np.logical_and(est_true, gt_true))
    TN = np.sum(np.logical_and(est_false, gt_false))
    FP = np.sum(np.logical_and(est_true, gt_false))
    FN = np.sum(np.logical_and(est_false, gt_true))
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    f1_score = 2*precision*recall / (precision+recall)

    return accuracy, precision, recall, f1_score

if __name__ == '__main__':
    set_seed(123)
    train_and_test()


