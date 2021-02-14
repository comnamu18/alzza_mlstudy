import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torchvision import transforms
import torchvision.transforms.functional as TF

import os
from PIL import Image


class AbaloneDataset(Dataset):
    def __init__(self, csv):
        self.csv = csv

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.csv.iloc[idx]["Rings"]
        label = label.astype(np.float32)

        sex = self.csv.iloc[idx]["Sex"]
        value = np.zeros([10], dtype=np.float32)

        if sex == "I":
            value[0] = 1.0
        if sex == "M":
            value[1] = 1.0
        if sex == "F":
            value[2] = 1.0

        value[3:] = self.csv.iloc[idx][1:-1]

        return {"label": label, "value": value}

    def __len__(self):
        return len(self.csv)


def extend_data(csv):
    csv.sort_values(csv.columns.to_list()[-1:], axis=0, inplace=True, ignore_index=True)
    star_cnt = 0

    for i in range(len(csv)):
        if csv.iloc[i][-1] > 0:
            break
        star_cnt += 1

    pulsar_cnt = len(csv) - star_cnt

    for idx in range(star_cnt - pulsar_cnt):
        csv = csv.append(csv.iloc[star_cnt + idx % pulsar_cnt], ignore_index=True)

    return csv


class PulsarDataset(Dataset):
    def __init__(self, csv):
        self.csv = csv

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = torch.tensor(self.csv.iloc[idx][-1], dtype=torch.float32)
        value = torch.tensor(self.csv.iloc[idx][:-1].to_list(), dtype=torch.float32)

        return {"label": label, "value": value}

    def __len__(self):
        return len(self.csv)


class SteelPlateDataset(Dataset):
    def __init__(self, csv):
        self.csv = csv

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = torch.tensor(self.csv.iloc[idx][-7:].to_list(), dtype=torch.float32)
        value = torch.tensor(self.csv.iloc[idx][:-7].to_list(), dtype=torch.float32)

        return {"label": label, "value": value}

    def __len__(self):
        return len(self.csv)


class FlowerDataset(Dataset):
    def __init__(self, resolution=(100, 100), input_shape=(-1,), transform=None):
        base_dir = "../chap05/flowers"
        self.resolution = resolution
        self.input_shape = input_shape
        self.output_shape = 1

        self.label = []
        self.img = []

        self.class_map = {}

        for c_id, c in enumerate(self._listdir(base_dir)):
            self.class_map[c_id] = c
            for d in os.listdir(os.path.join(base_dir, c)):
                if d[-3:] != "jpg":
                    continue

                read_img = Image.open(os.path.join(base_dir, c, d))
                read_img = read_img.resize(self.resolution)

                read_img = transforms.ToTensor()(read_img)
                read_img = torch.flatten(read_img)

                self.img.append(read_img)
                self.label.append(int(c_id))

        # print(self.class_map)
        # print(self.class_map[0])

    def __getitem__(self, idx):
        return {"x": self.img[idx], "y": self.label[idx]}

    def __len__(self):
        return len(self.img)

    def _listdir(self, path):
        return sorted(os.listdir(path))

    def get_class_name(self, class_id):
        return self.class_map[class_id]


class Office31Dataset(Dataset):
    def __init__(self, resolution=(100, 100), input_shape=(-1,)):
        super(Office31Dataset, self).__init__()
        base_dir = "../chap06/office31"
        self.resolution = resolution

        self.label = []
        self.img = []
        self.label0 = []
        self.label1 = []

        self.domain_map = {}
        self.class_map = {}
        self.domain_cnt = 3
        self.class_cnt = 31

        for d_id, domain in enumerate(self._listdir(base_dir)):
            self.domain_map[d_id] = domain
            domain_dir = os.path.join(base_dir, domain, "images")
            for c_id, c in enumerate(self._listdir(domain_dir)):
                self.class_map[c_id] = c
                class_dir = os.path.join(domain_dir, c)
                for img in self._listdir(class_dir):
                    if img[-4:] != ".jpg":
                        continue

                    read_img = Image.open(os.path.join(class_dir, img))
                    read_img = read_img.resize(self.resolution)

                    # read_img = torch.Tensor(np.array(read_img))
                    read_img = transforms.ToTensor()(read_img)
                    read_img = torch.flatten(read_img)

                    self.label0.append(d_id)
                    self.label1.append(c_id)
                    self.img.append(read_img)

        self.map = [self.domain_map, self.class_map]

    def __getitem__(self, idx):
        return {
            "x": self.img[idx],
            "domain": self.label0[idx],
            "category": self.label1[idx],
        }

    def __len__(self):
        return len(self.img)

    def _listdir(self, path):
        return sorted(os.listdir(path))

    # def _onehot(self, c_id, length):
    #     return np.eye(length)[np.array(c_id).astype(int)]

    def get_domain_name(self, class_id):
        return self.domain_map[class_id]

    def get_class_name(self, class_id):
        return self.class_map[class_id]


if __name__ == "__main__":
    import random

    fdata = Office31Dataset()
    n_fdata = len(fdata)
    split = int(n_fdata * 0.8)
    indices = list(range(n_fdata))

    train_sampler = SubsetRandomSampler(indices[:split])
    validation_sampler = SubsetRandomSampler(indices[split:])

    train_loader = DataLoader(fdata, batch_size=1, sampler=train_sampler)

    for data in train_loader:
        print(data["x"])
        break
