import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from torch.optim import SGD
import torch.nn.functional as F
from torchvision import transforms
import random

import sys

sys.path.append("../utils_torch")
from utils import *
from dataloader import *
from network import *
from config import *


def train_and_test(model, odata, train_loader, validation_loader, args):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for e in range(args.epoch):
        train_metric0 = 0
        train_metric1 = 0
        total = 0
        train_loss = []
        model.train()
        for i, data in enumerate(train_loader):
            x = data["x"].cuda()
            y0 = data["domain"].cuda()
            y1 = data["category"].cuda()

            pred = model(x)
            pred0 = pred[:, : odata.domain_cnt]
            pred1 = pred[:, odata.domain_cnt :]

            loss0 = criterion(pred0, y0)
            loss1 = criterion(pred1, y0)
            loss = loss0 + loss1

            train_loss.append(loss.item())

            train_metric0 += (
                torch.where(torch.argmax(pred0, axis=1) == y0, 1, 0).sum().item()
            )
            train_metric1 += (
                torch.where(torch.argmax(pred1, axis=1) == y1, 1, 0).sum().item()
            )
            total += y0.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = np.mean(train_loss)
        train_metric0 /= total
        train_metric1 /= total
        train_metric = (
            "{:.3f}".format(train_metric0) + "+" + "{:.3f}".format(train_metric1)
        )

        val_metric0 = 0
        val_metric1 = 0
        total = 0
        val_loss = []
        model.eval()
        for i, data in enumerate(validation_loader):
            x = data["x"].cuda()
            y0 = data["domain"].cuda()
            y1 = data["category"].cuda()

            pred = model(x)
            pred0 = pred[:, : odata.domain_cnt]
            pred1 = pred[:, odata.domain_cnt :]

            loss0 = criterion(pred0, y0)
            loss1 = criterion(pred1, y0)
            loss = loss0 + loss1

            val_loss.append(loss.item())

            val_metric0 += (
                torch.where(torch.argmax(pred0, axis=1) == y0, 1, 0).sum().item()
            )
            val_metric1 += (
                torch.where(torch.argmax(pred1, axis=1) == y1, 1, 0).sum().item()
            )
            total += y0.size(0)

        val_loss = np.mean(val_loss)
        val_metric0 /= total
        val_metric1 /= total
        val_metric = "{:.3f}".format(val_metric0) + "+" + "{:.3f}".format(val_metric1)

        if (e + 1) % args.report == 0:
            report(e, train_loss, train_metric, val_loss, val_metric)


def visualize(model, odata, idx):
    sample = odata[idx]
    x = sample["x"].cuda()
    y0 = sample["domain"]
    y1 = sample["category"]

    pred = model(x)
    pred = pred.detach().cpu()
    pred = pred.view(1, -1)
    pred0 = pred[:, : odata.domain_cnt]
    pred1 = pred[:, odata.domain_cnt :]
    domain = torch.argmax(pred0).item()
    category = torch.argmax(pred1).item()

    iscorrect0 = "O" if y0 == domain else "X"
    iscorrect1 = "O" if y1 == category else "X"

    print(f"Sample {idx}")
    print(
        f"Domain 추정확률 분포 {F.softmax(pred0).numpy()} => 추청 {odata.get_domain_name(domain)} : 정답 {odata.get_domain_name(y0)} => {iscorrect0}"
    )
    print(
        f"Class  추정확률 분포 {F.softmax(pred1).numpy()} => 추청 {odata.get_class_name(category)} : 정답 {odata.get_class_name(y1)} => {iscorrect1}"
    )
    print("")


if __name__ == "__main__":
    args = office31_mlp_config()
    set_seed(args.seed)

    odata = Office31Dataset()
    n_odata = len(odata)
    split = int(n_odata * 0.8)
    indices = list(range(n_odata))
    random.shuffle(indices)

    train_sampler = SubsetRandomSampler(indices[:split])
    validation_sampler = SubsetRandomSampler(indices[split:])

    train_loader = DataLoader(odata, batch_size=args.batchsize, sampler=train_sampler)
    validation_loader = DataLoader(
        odata, batch_size=args.batchsize, sampler=validation_sampler
    )

    model = mlp(args).cuda()
    summary(model, input_size=(args.in_features,))

    train_and_test(model, odata, train_loader, validation_loader, args)

    if args.visualize:
        print("\n", "#" * 20, "VISUALIZE", "#" * 20)
        for _ in range(args.visualize):
            idx = random.randint(0, n_odata)
            visualize(model, odata, idx)
