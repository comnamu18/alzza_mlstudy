import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

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
        
        label = torch.tensor(self.csv.iloc[idx][-7:].to_list, dtype=np.float32)
        _, label = torch.max(label, 0) 
        value = torch.tensor(self.csv.iloc[idx][:-7].to_list, dtype=np.float32)
        
        return {'label' : label, 'value' : value}
        
    def __len__(self):        
        return len(self.csv)


if __name__ == '__main__':
    csv = pd.read_csv("../chap02/pulsar_stars.csv")
    # data = PulsarDataset(csv)
    # print(len(data))

    data = PulsarDataset(csv, True)
    # print(len(data))