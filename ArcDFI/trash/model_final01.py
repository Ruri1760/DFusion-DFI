import torch
import torch.nn as nn
import pickle
from demo08.atom_extract import DFIFusionModel as model_two
from demo08.utils.ID_Fusion import cross_attention
from utils.multiAttention import MultiheadAttention, Attention
def open_files(file_name):
    with open("./pre_params/"+file_name, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

drug_dict_sw = open_files("drug_fea.pkl")
food_dict_sw = open_files("food_fea.pkl")
drug_food_ID = open_files("drug_food_ID.pkl")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EncoderLayer(nn.Module):
    def __init__(self, i_channel, o_channel, growth_rate, groups, pad2=7):
        super(EncoderLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=i_channel, out_channels=o_channel, kernel_size=(2 * pad2 + 1), stride=1,
                               groups=groups, padding=pad2,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(i_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=o_channel, out_channels=growth_rate, kernel_size=(2 * pad2 + 1), stride=1,
                               groups=groups, padding=pad2,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(o_channel)
        self.drop_rate = 0.1

    def forward(self, x):
        # xn = self.bn1(x)
        xn = self.relu(x)
        xn = self.conv1(xn)
        xn = self.bn2(xn)
        xn = self.relu(xn)
        xn = self.conv2(xn)

        return torch.cat([x, xn], 1)
    
class Encoder(nn.Module):
    def __init__(self, inc, outc, growth_rate, layers, groups, pad1=15, pad2=7):
        super(Encoder, self).__init__()
        self.layers = layers
        self.relu = nn.ReLU(inplace=True)
        self.conv_in = nn.Conv1d(in_channels=inc, out_channels=inc, kernel_size=(pad1 * 2 + 1), stride=1, padding=pad1,
                                 bias=False)
        self.dense_cnn = nn.ModuleList(
            [EncoderLayer(inc + growth_rate * i_la, inc + (growth_rate // 2) * i_la, growth_rate, groups, pad2) for i_la
             in
             range(layers)])
        self.conv_out = nn.Conv1d(in_channels=inc + growth_rate * layers, out_channels=outc, kernel_size=(pad1 * 2 + 1),
                                  stride=1,
                                  padding=pad1, bias=False)

    def forward(self, x):
        x = self.conv_in(x)
        for i in range(self.layers):
            x = self.dense_cnn[i](x)
        x = self.relu(x)
        x = self.conv_out(x)
        x = self.relu(x)
        return x
    
class fusion_model(nn.Module):
    def __init__(self, hidden1 = 256,attn_heads=1):
        super(fusion_model, self).__init__()

        self.hidden1 = hidden1       
        self.layers  = 1
        self.encoder_drug = Encoder(768, hidden1, 128, 3, groups=32, pad1=7, pad2=3)
        self.encoder_food = Encoder(768, hidden1, 128, 3, groups=32, pad1=7, pad2=3)
        self.ID_embedding = nn.Embedding(len(drug_food_ID), hidden1*2)
        self.category_embed = nn.Embedding(2, hidden1)
        self.label_embed = nn.Embedding(2, hidden1)
        self.drug_multihead = nn.ModuleList([MultiheadAttention(1, 256, 0.05) for _ in range(self.layers)])
        self.food_multihead = nn.ModuleList([MultiheadAttention(1, 256, 0.05) for _ in range(self.layers)])
        self.DF_Multihead = MultiheadAttention(2, 512, 0.05)

        self.model_two = model_two()
        #self.Informer_blocks = nn.ModuleList(
        #    [AttentionLayer(ProbAttention(None, 3, 0), hidden1, attn_heads),
        #     AttentionLayer(ProbAttention(None, 5, 0), hidden1, attn_heads)])
        self.cross_attention = cross_attention(512,512)
        self.adap_pool = nn.AdaptiveAvgPool1d(1)
        self.encoder_DFI = nn.Sequential(nn.Linear(512, 32), nn.BatchNorm1d(100), nn.Dropout(0.05), nn.ReLU(True))
        self.avg_pool = nn.ModuleList([nn.AdaptiveAvgPool1d(output_size=128),nn.AdaptiveAvgPool1d(output_size=128)])
        self.drop_out = nn.ModuleList([nn.Dropout(p=0.1),nn.Dropout(p=0.05)])
        self.DFI_feature = nn.ModuleList([nn.Linear(3200, 512), nn.Linear(512, 128)])
        self.Batch = nn.ModuleList([nn.LayerNorm(512), nn.LayerNorm(128)])
        self.act = nn.GELU()
        self.DFI_Pre = nn.Linear(128, 2)
    def pack(self, data_batch):

        drug_list, food_list = data_batch['drug_id'], data_batch["food_id"]
        N = len(drug_list)
        max_len = 100
        drug_fea = torch.zeros((N, max_len, 768)).to(device) 
        food_fea = torch.zeros((N, max_len, 768)).to(device)
        drug_IDs, food_IDs = [],[]
        for i in range(N):
            if food_list[i] in food_dict_sw:
                temp = food_dict_sw[food_list[i]]
                
                if temp.shape[0]<=max_len:
                    food_fea[i, :temp.shape[0],:] = torch.tensor(temp).to(device)
                else:
                    food_fea[i, :,:] = torch.tensor(temp)[:max_len,:].to(device)

            if drug_list[i] in drug_dict_sw:
                temp = drug_dict_sw[drug_list[i]]
                drug_len = temp.shape[0]
                
                if drug_len<=max_len:
                    drug_fea[i,:drug_len,:] = torch.tensor(temp).to(device)
                else:
                    drug_fea[i,:,:]= torch.tensor(temp)[ :max_len,:].to(device)
            drug_IDs.append(drug_food_ID[drug_list[i]])
            food_IDs.append(drug_food_ID[food_list[i]])
        drug_IDs = torch.tensor(drug_IDs).to(torch.int).to(device)
        food_IDs = torch.tensor(food_IDs).to(torch.int).to(device)

        drug_IDs = self.ID_embedding(drug_IDs).unsqueeze(1).repeat(1, 100, 1)
        food_IDs = self.ID_embedding(food_IDs).unsqueeze(1).repeat(1, 100, 1)

        labels = torch.tensor([0, 1]).to(device)

        label_embeds = self.category_embed(labels)
        embed_0 = label_embeds[0].unsqueeze(0).unsqueeze(0)  # [1, 1, embed_dim]
        embed_0 = embed_0.repeat(N, max_len, 1)      # [batch_size, 100, embed_dim]
        embed_1 = label_embeds[1].unsqueeze(0).unsqueeze(0)  # [1, 1, embed_dim]
        embed_1 = embed_1.repeat(N, max_len, 1) 

        return drug_fea.contiguous(), food_fea.contiguous(), drug_IDs.contiguous(), food_IDs.contiguous(), embed_0.contiguous(), embed_1.contiguous()
    
    def forward(self,dataset_batch):

        drug_fea,food_fea, drug_IDs, food_IDs, embed_0, embed_1 = self.pack(dataset_batch)

        food_fea = self.encoder_food(food_fea.permute(0, 2, 1)).permute(0, 2, 1) # size(N,100,256)
        drug_fea = self.encoder_drug(drug_fea.permute(0, 2, 1)).permute(0, 2, 1)

        for i in range(self.layers):
            food_fea, food_attn_weights = self.drug_multihead[i]( food_fea, food_fea, food_fea)
            drug_fea, drug_attn_weights = self.food_multihead[i]( drug_fea, drug_fea, drug_fea)

        temp = food_fea + drug_fea
        drug_fea = self.avg_pool[0](drug_fea)
        food_fea = self.avg_pool[1](food_fea)
        DF_Feature = torch.cat((drug_fea, food_fea, temp), 2)
        DF_Feature_one,_ = self.DF_Multihead(DF_Feature, DF_Feature, DF_Feature)
        #DF_Feature_one = self.adap_pool(DF_Feature_one.permute(0,2,1)).permute(0, 2, 1)
        DF_Feature_two,a,b = self.model_two(dataset_batch)
        
        #print(DF_Feature_one.shape, DF_Feature_two.shape)
        DF_Feature_two = DF_Feature_two.repeat(1, 100, 1) 


        DF_Feature = self.cross_attention(DF_Feature_one, DF_Feature_two, drug_IDs+food_IDs)
        #print(DF_Feature.shape)
        DF_Feature = self.encoder_DFI(DF_Feature)
        DF_Feature = DF_Feature.view(DF_Feature.shape[0], -1) 

        for i in range(2):
            DF_Feature = self.drop_out[i](self.act(self.Batch[i](self.DFI_feature[i](DF_Feature))))
        DF_Feature = self.DFI_Pre(DF_Feature )

        return DF_Feature
        #return DF_Feature,drug_attn_weights, food_attn_weights, a,b