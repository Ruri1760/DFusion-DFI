import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
device = "cuda" if torch.cuda.is_available() else'cpu'


class DrugFoodDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.pairs = dataframe[['drugcompound_id', 'foodcompound_id', 'dfi_label']].values
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        drug_id, food_id, label = self.pairs[idx]
        
        return {
            'drug_id': drug_id,
            'food_id': food_id,
            'label': torch.tensor(label, dtype=torch.long)
        }


def create_data_loaders2(csv_path, batch_size=32, flag=1, shuffle=True):
    df = pd.read_csv(csv_path)  
    split_flag = ["split","split_newdrug","split_newfood"]
    print("split_flag:",split_flag[flag] )
    train_df = df[df[split_flag[flag]] == 'train']
    val_df = df[df[split_flag[flag]] == 'valid']
    test_df = df[df[split_flag[flag]] == 'test']

    train_dataset = DrugFoodDataset(train_df)
    val_dataset = DrugFoodDataset(val_df)
    test_dataset = DrugFoodDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

