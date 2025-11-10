import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import rdchem
import numpy as np
from utils.multiAttention import Attention

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"use device: {device}")


def open_files(file_name):
    with open("./pre_params/"+file_name, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

drug_atom_fea = open_files("drug_all_atom_fea.pkl")
drug_smiles   = open_files("drug_id2smiles.pkl")
drug_food_ID  = open_files("drug_food_ID.pkl")

def get_bond_feature(bond):

    bond_type = bond.GetBondType()
    bond_type_feat = [
        1.0 if bond_type == rdchem.BondType.SINGLE else 0.0,
        1.0 if bond_type == rdchem.BondType.DOUBLE else 0.0,
        1.0 if bond_type == rdchem.BondType.TRIPLE else 0.0,
        1.0 if bond_type == rdchem.BondType.AROMATIC else 0.0
    ]
    conjugate_feat = [1.0 if bond.GetIsConjugated() else 0.0]
    ring_feat = [1.0 if bond.IsInRing() else 0.0]
    return np.array(bond_type_feat + conjugate_feat + ring_feat, dtype=np.float32)


class GATEncoder(nn.Module):
    def __init__(self, input_dim=55, edge_dim=6, hidden_dim=256, output_dim=256, heads=4):
        super().__init__()
        self.conv1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=True, 
            edge_dim=edge_dim,
            dropout=0.1, 
            add_self_loops=True 
        )

        self.conv2 = GATConv(
            in_channels=hidden_dim * heads, 
            out_channels=output_dim,
            heads=1,
            concat=False,
            edge_dim=edge_dim,
            dropout=0.1
        )

        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):

        edge_index = edge_index.to(torch.long)

        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)  
        x = self.relu(x)
        
        node_feats = self.conv2(x, edge_index, edge_attr)
        node_feats = self.norm2(node_feats)
        
        return node_feats

class GATExtractPart(nn.Module):
    def __init__(self, drug_input_dim=55, food_input_dim=55, edge_dim=6, hidden_dim=256, output_dim=256):
        super().__init__()
        self.drug_gat = GATEncoder(drug_input_dim, edge_dim, hidden_dim, output_dim).to(device)
        self.attention1 = Attention( hidden_dim)
        self.food_embedding =  nn.Embedding(29, 256)
        self.adap_pool = nn.AdaptiveMaxPool1d(1)

    def pack(self, data_batch):
        data_batch,_ = data_batch
        drug_list, food_list = data_batch[0], data_batch[1]
        drug_data_list, drug_len_list= [], []
        affinity_list = []

        for drug_id in drug_list:
            smiles = drug_smiles.get(drug_id, "")
            atom_feats = drug_atom_fea.get(drug_id, {}).get("atom_fea", None)
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            
            if mol is None or atom_feats is None or atom_feats.shape[0] != mol.GetNumAtoms():
                drug_len_list.append(0)
                continue
            
            x = atom_feats.clone().detach().to(device, dtype=torch.float32)
            drug_len_list.append(mol.GetNumAtoms())
            
            edges, edge_attrs = [], []
            for bond in mol.GetBonds():
                u = bond.GetBeginAtomIdx()
                v = bond.GetEndAtomIdx()
                edges.extend([(u, v), (v, u)])
                bond_feat = get_bond_feature(bond)
                edge_attrs.extend([bond_feat, bond_feat])
            
            edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous() if edges else torch.empty((2, 0), device=device)
            if edge_attrs:

                edge_attr_np = np.array(edge_attrs)  
                edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32, device=device)
            else:
                edge_attr = torch.empty((0, 6), device=device)
            drug_data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

        food_IDs = []
        for food_ID in food_list:
            food_IDs.append(drug_food_ID[food_ID])
        food_fea = torch.tensor([x - 676 for x in food_IDs], dtype=torch.int).to(device)
        
        
        return drug_data_list, food_fea, drug_len_list

    def forward(self, dataset_batch):
        drug_data, food_fea, drug_len = self.pack(dataset_batch)
        batch_size = len(drug_len)
        drug_attn_list = []
        drug_fea = torch.zeros((batch_size, 1, 256), device=device)
        if drug_data:
            drug_batch = Batch.from_data_list(drug_data).to(device)
            drug_node_feats = self.drug_gat(drug_batch.x, drug_batch.edge_index, drug_batch.edge_attr)
            
            d_interval = 0
            for i in range(batch_size):
                d_len = drug_len[i]
                if d_len <= 0 or d_interval + d_len > drug_node_feats.shape[0]:
                    continue

                d_node_feat = drug_node_feats[d_interval:d_interval + d_len]
                d_node_feat, d_attn = self.attention1(d_node_feat)
                drug_attn_list.append(d_attn)


                d_node_feat_permuted = d_node_feat.permute(1, 0)
                d_graph_feat = self.adap_pool(d_node_feat_permuted).squeeze(-1) 
                drug_fea[i] = d_graph_feat.unsqueeze(0) 
                
                d_interval += d_len

        food_fea = self.food_embedding(food_fea).unsqueeze(1)


        
        return drug_fea, food_fea, drug_attn_list

class ImprovedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "It is mandatory that embed_dim is an integer multiple of num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)  
        
        nn.init.xavier_uniform_(self.q_proj.weight) 
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        nn.init.constant_(self.q_proj.bias, 0.1)
        nn.init.constant_(self.k_proj.bias, 0.1)
        nn.init.constant_(self.v_proj.bias, 0.1)
        nn.init.constant_(self.out_proj.bias, 0.1)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim) 

    def forward(self, q, k, v, key_padding_mask=None):
        batch_size = q.size(0)
        
        q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))  
        attn_scores = attn_scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=device))
        
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(1).to(device)
            attn_scores = attn_scores.masked_fill(mask, -1e9)  
        
        attn_weights = F.softmax(attn_scores, dim=-1) 
        attn_weights = self.dropout(attn_weights)  
        
        attn_output = torch.matmul(attn_weights, v)  
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        attn_output = self.out_proj(attn_output) 
        
        output = self.norm(attn_output + q.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim))
        
        return output, attn_weights
    

class structural_feature_extractor(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_classes=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes 

        self.graph_extractor = GATExtractPart(
            drug_input_dim=55, food_input_dim=55, edge_dim=6,
            hidden_dim=embed_dim, output_dim=embed_dim
        ).to(device)
        

        self.inter_attn = ImprovedMultiheadAttention(embed_dim * 2, num_heads, dropout)  # 输入=药物+食物特征


    def forward(self, dataset_batch):

        drug_fea, food_fea, drug_attn_list = self.graph_extractor(dataset_batch)


        inter_feat = torch.cat([drug_fea, food_fea], dim=-1)

        inter_fused_fea, inter_attn = self.inter_attn(inter_feat, inter_feat, inter_feat)

        return inter_fused_fea, drug_attn_list