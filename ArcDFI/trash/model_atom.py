import torch
import torch.nn as nn
import pickle
from utils.multiAttention import MultiheadAttention
def open_files(file_name):
    with open("./pre_params/"+file_name, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

drug_atom_fea = open_files("drug_atom_fea.pkl")
food_atom_fea = open_files("food_atom_fea.pkl")
drug_food_ID = open_files("drug_food_ID.pkl")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
class fusion_model(nn.Module):
    def __init__(self, hidden1 = 256,attn_heads=1):
        super(fusion_model, self).__init__()

        self.hidden1 = hidden1       
        self.layers  = 1
        self.encoder_drug = nn.Sequential(nn.Linear(55, 256), nn.BatchNorm1d(50), nn.Dropout(0.2), nn.ReLU(True))
        self.encoder_food = nn.Sequential(nn.Linear(55, 256), nn.BatchNorm1d(50), nn.Dropout(0.2), nn.ReLU(True))
        self.ID_embedding = nn.Embedding(len(drug_food_ID), hidden1)
        self.category_embed = nn.Embedding(2, hidden1)
        self.label_embed = nn.Embedding(2, hidden1)
        self.drug_multihead = nn.ModuleList([MultiheadAttention(1, 256, 0.05) for _ in range(self.layers)])
        self.food_multihead = nn.ModuleList([MultiheadAttention(1, 256, 0.05) for _ in range(self.layers)])
        self.DF_Multihead = MultiheadAttention(2, 512, 0.05)
        #self.Informer_blocks = nn.ModuleList(
        #    [AttentionLayer(ProbAttention(None, 3, 0), hidden1, attn_heads),
        #     AttentionLayer(ProbAttention(None, 5, 0), hidden1, attn_heads)])
        self.avg_pool = nn.ModuleList([nn.AdaptiveAvgPool1d(output_size=128),nn.AdaptiveAvgPool1d(output_size=128)])
        self.encoder_DFI = nn.Sequential(nn.Linear(512, 32), nn.BatchNorm1d(50), nn.Dropout(0.05), nn.ReLU(True))
        self.drop_out = nn.ModuleList([nn.Dropout(p=0.1),nn.Dropout(p=0.05)])
        self.DFI_feature = nn.ModuleList([nn.Linear(1600, 1024),nn.Linear(1024, 128)])
        self.Batch = nn.ModuleList([nn.LayerNorm(1024),nn.LayerNorm(128)])
        self.act = nn.GELU()
        self.DFI_Pre = nn.Linear(128, 2)

    def pack(self, data_batch):

        drug_list, food_list = data_batch['drug_id'], data_batch["food_id"]
        N = len(drug_list)
        max_len = 50
        drug_fea = torch.zeros((N, max_len, 55)).to(device) 
        food_fea = torch.zeros((N, max_len, 55)).to(device)
        drug_IDs, food_IDs = [],[]
        for i in range(N):
            if food_list[i] in food_atom_fea.keys():
                temp1 = food_atom_fea[food_list[i]]["atom_fea"]
                if temp1.shape[0]<=max_len:
                    food_fea[i, :temp1.shape[0],:] = temp1.to(device)
                else:
                    food_fea[i, :,:] = temp1[:max_len,:].to(device)

            if drug_list[i] in drug_atom_fea.keys():
                temp2 = drug_atom_fea[drug_list[i]]["atom_fea"]
                drug_len = temp2.shape[0]
                if drug_len<=max_len:
                    drug_fea[i,:drug_len,:] = temp2.to(device)
                else:
                    drug_fea[i,:,:]= temp2[ :max_len,:].to(device)
            drug_IDs.append(drug_food_ID[drug_list[i]])
            food_IDs.append(drug_food_ID[food_list[i]])
        drug_IDs = torch.tensor(drug_IDs).to(torch.int).to(device)
        food_IDs = torch.tensor(food_IDs).to(torch.int).to(device)

        drug_IDs = self.ID_embedding(drug_IDs).unsqueeze(1).repeat(1, max_len, 1)
        food_IDs = self.ID_embedding(food_IDs).unsqueeze(1).repeat(1, max_len, 1)

        labels = torch.tensor([0, 1]).to(device)

        label_embeds = self.category_embed(labels)
        embed_0 = label_embeds[0].unsqueeze(0).unsqueeze(0)  # [1, 1, embed_dim]
        embed_0 = embed_0.repeat(N, max_len, 1)      # [batch_size, 100, embed_dim]
        embed_1 = label_embeds[1].unsqueeze(0).unsqueeze(0)  # [1, 1, embed_dim]
        embed_1 = embed_1.repeat(N, max_len, 1) 

        return drug_fea.contiguous(), food_fea.contiguous(), drug_IDs.contiguous(), food_IDs.contiguous(), embed_0.contiguous(), embed_1.contiguous()
    
    def class_imbalance2(self, DFI_fea, labels, label_num=2):
        device = DFI_fea.device
        batch_size = DFI_fea.shape[0]
    
        # 1. 生成标签嵌入向量（0和1对应的特征）
        label_ids = torch.tensor([0, 1], device=device)
        label_emb = self.label_embed(label_ids)  # 形状: [2, 256]
        label_0_emb = label_emb[0].unsqueeze(0)  # [1, 256]
        label_1_emb = label_emb[1].unsqueeze(0)  # [1, 256]
    
        # 2. 对药物和食物特征在第二个维度取平均 [batch_size, 200, 256] -> [batch_size, 256]
        DFI_fea = torch.mean(DFI_fea, dim=1)  # 平均后药物特征
    
        # 3. 统计标签0和1的数量及占比
        count_0 = (labels == 0).sum().float()
        count_1 = (labels == 1).sum().float()
        ratio_0 = count_0 / batch_size  # 标签0的数量占比
        ratio_1 = count_1 / batch_size  # 标签1的数量占比
        #ratio_0,ratio_1 =1,1
        mask_0 = (labels == 0)
        drug_0_loss = torch.tensor(0.0, device=device)
        if count_0 > 0:
            drug_0_samples = DFI_fea[mask_0]  # 筛选标签0的样本
            drug_0_dist = torch.norm(drug_0_samples - label_0_emb, dim=1)  # 欧氏距离
            drug_0_loss = drug_0_dist.mean()  # 标签0样本的平均损失
    
        # 3.2 计算标签1对应的药物损失
        mask_1 = (labels == 1)
        drug_1_loss = torch.tensor(0.0, device=device)
        if count_1 > 0:
            drug_1_samples = DFI_fea[mask_1]  # 筛选标签1的样本
            drug_1_dist = torch.norm(drug_1_samples - label_1_emb, dim=1)  # 欧氏距离
            drug_1_loss = drug_1_dist.mean()  # 标签1样本的平均损失
    
        # 3.3 药物特征的加权损失 (a部分)
        a = ratio_1 * drug_0_loss + ratio_0 * drug_1_loss

        return a
    def forward(self,dataset_batch):

        drug_fea,food_fea, drug_IDs, food_IDs, embed_0, embed_1 = self.pack(dataset_batch)

        food_fea = self.encoder_food(food_fea) # size(N,100,256)
        drug_fea = self.encoder_drug(drug_fea)

        for i in range(self.layers):
            food_fea, food_attn_weights = self.drug_multihead[i]( food_fea, food_fea, food_fea)
            drug_fea, drug_attn_weights = self.food_multihead[i]( drug_fea, drug_fea, drug_fea)

        temp = drug_fea + food_fea
        drug_fea = self.avg_pool[0](drug_fea)
        food_fea = self.avg_pool[1](food_fea)
        DF_Feature = torch.cat((drug_fea, food_fea, temp), 2)
        DF_Feature,_ = self.DF_Multihead(DF_Feature, DF_Feature, DF_Feature)
        
        DF_Feature = self.encoder_DFI(DF_Feature)
        DF_Feature = DF_Feature.view(DF_Feature.shape[0], -1) 
        for i in range(2):
            DF_Feature = self.drop_out[i](self.act(self.Batch[i](self.DFI_feature[i](DF_Feature))))
        DF_Feature = self.DFI_Pre(DF_Feature )

        return DF_Feature,0
        #return DF_Feature,self.class_imbalance2(temp, dataset_batch['label'].to(device))