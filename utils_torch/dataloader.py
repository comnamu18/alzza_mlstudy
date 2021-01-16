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