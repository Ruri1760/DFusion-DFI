import torch
import torch.nn as nn
import pickle
from utils.multiAttention import MultiheadAttention
from utils.multiAttention import Attention
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from rdkit import Chem
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def open_files(file_name):
    with open("./pre_params/"+file_name, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

# 加载预处理数据
drug_atom_fea = open_files("drug_all_atom_fea.pkl")
food_atom_fea = open_files("food_all_atom_fea.pkl")
drug_food_ID = open_files("drug_food_ID.pkl")
drug_smiles = open_files("drug_id2smiles.pkl")
food_smiles = open_files("food_id2smiles.pkl")

class GCNEncoder(nn.Module):
    """GCN编码器，用于提取分子图特征"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index):
        # 图卷积层
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        
        x = self.conv3(x, edge_index)
        return x

class GCN_extract_part(nn.Module):
    def __init__(self, drug_input_dim, food_input_dim, hidden_dim=256, output_dim=256, attn_heads=4):
        super(GCN_extract_part, self).__init__()
        
        # 药物GCN编码器
        self.drug_gcn = GCNEncoder(drug_input_dim, hidden_dim, output_dim).to(device)
        # 食物GCN编码器
        self.food_gcn = GCNEncoder(food_input_dim, hidden_dim, output_dim).to(device)

        # 多头注意力机制，用于融合药物和食物特征
        self.attention1 = Attention( hidden_dim)
        self.attention2 = Attention( hidden_dim)
        
    def pack(self, data_batch):
        drug_list, food_list = data_batch['drug_id'], data_batch["food_id"]
        drug_data_list, food_data_list = [],[]
        drug_len_list, food_len_list = [],[]
        # 处理药物分子
        for drug_ID in drug_list:
            smiles = drug_smiles[drug_ID]
            try:
                mol = Chem.MolFromSmiles(smiles)
            except:
                mol = None
            if mol is None:
                drug_len_list.append(0)
                continue  # 跳过无效的SMILES
            atom_features = drug_atom_fea[drug_ID]["atom_fea"]
            drug_len_list.append(len(drug_atom_fea[drug_ID]["atom_seq"]))
            x = atom_features.clone().detach().to(device, dtype=torch.float)
            
            # 构建边索引
            edges = []
            for bond in mol.GetBonds():
                u = bond.GetBeginAtomIdx()
                v = bond.GetEndAtomIdx()
                edges.append((u, v))
                edges.append((v, u))  # 无向图
            
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
                
            data = Data(x=x, edge_index=edge_index)
            drug_data_list.append(data)
        
        # 处理食物分子
        for food_ID in food_list:
            smiles = food_smiles[food_ID]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue  # 跳过无效的SMILES
            atom_features = food_atom_fea[food_ID]["atom_fea"]
            food_len_list.append(len(food_atom_fea[food_ID]["atom_seq"]))
            x = atom_features.clone().detach().to(device, dtype=torch.float)
            
            # 构建边索引
            edges = []
            for bond in mol.GetBonds():
                u = bond.GetBeginAtomIdx()
                v = bond.GetEndAtomIdx()
                edges.append((u, v))
                edges.append((v, u))  # 无向图
            
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
                
            data = Data(x=x, edge_index=edge_index)
            food_data_list.append(data)
            
        return drug_data_list, food_data_list, drug_len_list, food_len_list
    
    
    def forward(self, dataset_batch):
        # 处理输入数据，转换为图数据结构
        drug_data_list, food_data_list, drug_len_list, food_len_list= self.pack(dataset_batch)  
        drug_batch = Batch.from_data_list(drug_data_list).to(device)
        food_batch = Batch.from_data_list(food_data_list).to(device)
        
        # 使用GCN提取节点特征
        drug_node_feats = self.drug_gcn(drug_batch.x, drug_batch.edge_index)
        food_node_feats = self.food_gcn(food_batch.x, food_batch.edge_index)
        drug_fea = torch.zeros(len(drug_len_list), 100, 256).to(device)
        food_fea = torch.zeros(len(food_len_list), 100, 256).to(device)
        interval = 0
        for i, steps in enumerate(drug_len_list):
            if steps != 0 :
                drug_temp= drug_node_feats[interval: interval+steps]
                interval += steps
                if steps <= 100:
                    drug_fea[i, :steps, :]  = drug_temp
                else:
                    drug_fea[i, :, :] = drug_temp[:100, :]
        interval = 0
        for i, steps in enumerate(food_len_list):
            food_temp = food_node_feats[interval: interval+steps]
            interval += steps
            if steps <= 100:
                food_fea[i, :steps, :]  = food_temp
            else:
                food_fea[i, :, :] = food_temp[:100, :]

        return drug_fea, food_fea, drug_len_list, food_len_list
    
class fusion_model(nn.Module):
    def __init__(self, hidden1 = 256,attn_heads=1):
        super(fusion_model, self).__init__()

        
        self.hidden1 = hidden1       
        self.layers  = 1
        self.GCN_extract_part = GCN_extract_part(55, 55)
        self.drug_multihead = nn.ModuleList([MultiheadAttention(1, 256, 0.05) for _ in range(self.layers)])
        self.food_multihead = nn.ModuleList([MultiheadAttention(1, 256, 0.05) for _ in range(self.layers)])
        self.DF_Multihead = MultiheadAttention(2, 512, 0.05)

        self.avg_pool = nn.ModuleList([nn.AdaptiveAvgPool1d(output_size=128),nn.AdaptiveAvgPool1d(output_size=128)])
        
    
    def forward(self,dataset_batch):
        #drug_len_list, food_len_list, drug_attention_list, food_attention_list
        drug_fea, food_fea, _,_,= self.GCN_extract_part(dataset_batch)
        for i in range(self.layers):
            food_fea, food_attn_weights = self.drug_multihead[i]( food_fea,  food_fea, food_fea)
            drug_fea, drug_attn_weights = self.food_multihead[i]( drug_fea,  drug_fea, drug_fea)
        temp = drug_fea + food_fea
        drug_fea = self.avg_pool[0](drug_fea)
        food_fea = self.avg_pool[1](food_fea)
        DF_Feature = torch.cat((drug_fea, food_fea, temp), 2)
        DF_Feature,_ = self.DF_Multihead(DF_Feature, DF_Feature, DF_Feature)
        return DF_Feature