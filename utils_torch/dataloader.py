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
        
        label = self.csv.iloc[idx]['Rings']
        label = label.astype(np.float32)
        
        sex = self.csv.iloc[idx]['Sex']
        value = np.zeros([10], dtype=np.float32)
        
        if sex == 'I': value[0] = 1.0
        if sex == 'M': value[1] = 1.0
        if sex == 'F': value[2] = 1.0
        
        value[3:] = self.csv.iloc[idx][1:-1]
        
        return {'label' : label, 'value' : value}
        
    def __len__(self):  
        return len(self.csv)


def extend_data(csv):
    csv.sort_values(csv.columns.to_list()[-1:], axis=0, inplace=True, ignore_index=True)
    star_cnt = 0

    for i in range(len(csv)):
        if csv.iloc[i][-1] > 0 :
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
    
        return {'label' : label, 'value' : value}
        
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
        
        return {'label' : label, 'value' : value}
        
    def __len__(self):        
        return len(self.csv)


class FlowerDataset(Dataset):
    def __init__(self, resolution=(100,100), input_shape=(-1,), transform=None):
        base_dir = '../chap05/flowers'
        self.resolution = resolution
        self.input_shape = input_shape
        self.output_shape = 1

        self.label = []
        self.img = []

        self.class_map = {}

        for c_id, c in enumerate(self._listdir(base_dir)):
            self.class_map[c_id] = c
            for d in os.listdir(os.path.join(base_dir, c)):
                if d[-3:] != 'jpg':
                    continue
                
                read_img = Image.open(os.path.join(base_dir, c, d))
                read_img = read_img.resize(self.resolution)

                read_img = transforms.ToTensor()(read_img)
                read_img = torch.flatten(read_img)

                self.img.append(read_img)
                self.label.append(int(c_id))

    def __getitem__(self, idx):
        return {'x': self.img[idx], 'y': self.label[idx]}
    
    def __len__(self):
        return len(self.img)
    
    def _listdir(self, path):
        return sorted(os.listdir(path))



if __name__ == '__main__':
    import random
    # csv = pd.read_csv("../chap03/faults.csv")
    fdata = FlowerDataset()
    n_fdata = len(fdata)
    split = int(n_fdata * 0.8)
    print(split)
    indices = list(range(n_fdata))
    # random.shuffle(indices)

    train_sampler = SubsetRandomSampler(indices[:split])
    validation_sampler = SubsetRandomSampler(indices[split:])
    
    train_loader = DataLoader(fdata, batch_size=8, sampler=train_sampler)

    for data in train_loader:
        print(data['y'])

    # print(list(dloader))

    # data = PulsarDataset(csv, True)
    # print(len(data))