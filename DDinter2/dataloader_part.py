import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
device = "cuda" if torch.cuda.is_available() else'cpu'
def open_files(file_name):
    with open("./pre_params/"+file_name, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

def judge_sample_isin(drug_ID, food_name, df):
    for i in range(len(df)):
        if drug_ID == df['Drug ID'][i] and food_name == df['Food name'][i]:
            return 1
    return 0

def deal_with_data(df):
    ID_list = df['Drug ID'].unique().tolist()
    food_name_list = df['Food name'].unique().tolist()
    print(len(ID_list),len(food_name_list))
    pos_dataset = []
    neg_dataset = []
    for drug_ID in ID_list:
        for food_name in food_name_list:
            label = judge_sample_isin(drug_ID, food_name, df)
            if label:
                pos_dataset.append([drug_ID, food_name, label])
            else:
                neg_dataset.append([drug_ID, food_name, label])
    return pos_dataset, neg_dataset

def train_test_data1(data_lis):
    data_lis = np.array(data_lis)
    drug_pair = data_lis[:, :2]
    Y = data_lis[:, 2].astype(int)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

    train_X_data = []
    train_Y_data = []
    test_X_data = []
    test_Y_data = []

    for train, test in kfold.split(drug_pair, Y):
        train_X_data.append(drug_pair[train])
        train_Y_data.append(Y[train])
        test_X_data.append(drug_pair[test])
        test_Y_data.append(Y[test])

    return train_X_data, train_Y_data, test_X_data, test_Y_data

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].tolist(), self.Y[idx]

def get_dataloader(trainx, trainy, testx, testy, batch_size=64):
    train_dataset = CustomDataset(trainx, trainy)
    test_dataset = CustomDataset(testx, testy)

    loader_train = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    print(len(trainx)+len(testx))
    return loader_train, loader_test


Mechanism_dict = {
    "Synergy":0,
    "Metabolism":1,
    "Absorption":2,
    "Others":3,
    "Unknown":3,
    "Antagonism":3,
    "Excretion":3
}

def deal_with_data_mechanism(df):
    datasets = []
    ID_list = df['Drug ID'].unique().tolist()
    food_name_list = df['Food name'].unique().tolist()
    print(len(ID_list),len(food_name_list))
    for i in range(len(df)):
        drug_ID = df['Drug ID'][i]
        food_name = df["Food name"][i]
        label = Mechanism_dict[df["Mechanism"][i]]
        datasets.append([drug_ID, food_name, label])

    return datasets
