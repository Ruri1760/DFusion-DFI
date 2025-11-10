import torch
import torch.nn as nn
import pickle
from atom_extract import structural_feature_extractor
from utils.ID_Fusion import ID_Fusion
from utils.multiAttention import MultiheadAttention
def open_files(file_name):
    with open("./pre_params/"+file_name, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

drug_dict_sw = open_files("drug_txt_fea.pkl")
food_dict_sw = open_files("food_txt_fea.pkl")
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
    
class DFusion_DFI_model(nn.Module):
    def __init__(self, hidden1 = 256, out_dim=2):
        super(DFusion_DFI_model, self).__init__()

        self.hidden1 = hidden1       
        self.layers  = 1
        self.encoder_drug = Encoder(1536, hidden1, 128, 3, groups=32, pad1=7, pad2=3)
        self.encoder_food = Encoder(1536, hidden1, 128, 3, groups=64, pad1=7, pad2=3)
        self.ID_embedding = nn.Embedding(len(drug_food_ID), hidden1*2)

        self.drug_multihead = nn.ModuleList([MultiheadAttention(1, 256, 0.1) for _ in range(self.layers)])
        self.food_multihead = nn.ModuleList([MultiheadAttention(1, 256, 0.1) for _ in range(self.layers)])
        self.DF_Multihead = MultiheadAttention(2, 512, 0.05)
        self.adap_pool = nn.AdaptiveAvgPool1d(1)

        self.model_two = structural_feature_extractor()
        self.Fusion = ID_Fusion(512,512)
        
        self.avg_pool = nn.ModuleList([nn.AdaptiveAvgPool1d(output_size=128),nn.AdaptiveAvgPool1d(output_size=128)])
        self.drop_out = nn.ModuleList([nn.Dropout(p=0.1),nn.Dropout(p=0.05)])
        self.DFI_feature = nn.ModuleList([nn.Linear(512, 512), nn.Linear(512, 128)])
        self.Batch = nn.ModuleList([nn.LayerNorm(512), nn.LayerNorm(128)])
        self.act = nn.GELU()
        self.DFI_Pre = nn.Linear(128, out_dim)
    def pack(self, data_batch):
        data_batch,y = data_batch
        drug_list, food_list = data_batch[0], data_batch[1]
        
        N = len(drug_list)
        max_len = 200
        drug_fea = torch.zeros((N, max_len, 1536)).to(device) 
        food_fea = torch.zeros((N, max_len, 1536)).to(device)
        drug_IDs, food_IDs = [],[]
        for i in range(N):
            if food_list[i] in food_dict_sw:
                temp1 = torch.tensor(food_dict_sw[food_list[i]]['text_fea'])
                if temp1.ndim == 3:
                    temp1 = temp1.squeeze(0)
                if temp1.shape[0]<=max_len:
                    food_fea[i, :temp1.shape[0],:] = temp1.to(device)
                else:
                    food_fea[i, :,:] = temp1[:max_len,:].to(device)

            if drug_list[i] in drug_dict_sw:
                temp2 = torch.tensor(drug_dict_sw[drug_list[i]]['text_fea'])
                if temp2.ndim == 3:
                    temp2 = temp2.squeeze(0)
                drug_len = temp2.shape[0]
                if drug_len<=max_len:
                    drug_fea[i,:drug_len,:] = temp2.to(device)
                else:
                    drug_fea[i,:,:]= temp2[:max_len,:].to(device)
            drug_IDs.append(drug_food_ID[drug_list[i]])
            food_IDs.append(drug_food_ID[food_list[i]])
        drug_IDs = torch.tensor(drug_IDs).to(torch.int).to(device)
        food_IDs = torch.tensor(food_IDs).to(torch.int).to(device)

        drug_IDs = self.ID_embedding(drug_IDs).unsqueeze(1)
        food_IDs = self.ID_embedding(food_IDs).unsqueeze(1)

        return drug_fea.contiguous(), food_fea.contiguous(), drug_IDs.contiguous(), food_IDs.contiguous()
    
    def forward(self,dataset_batch):
        ###############################Textual Feature Extractor######################################
        drug_fea,food_fea, drug_IDs, food_IDs= self.pack(dataset_batch)
        _,labels = dataset_batch
        food_fea = self.encoder_food(food_fea.permute(0, 2, 1)).permute(0, 2, 1) # size(N,100,256)
        drug_fea = self.encoder_drug(drug_fea.permute(0, 2, 1)).permute(0, 2, 1)

        for i in range(self.layers):
            food_fea, food_attn_weights = self.drug_multihead[i]( food_fea, food_fea, food_fea)
            drug_fea, drug_attn_weights = self.food_multihead[i]( drug_fea, drug_fea, drug_fea)

        temp = food_fea+drug_fea
        drug_fea = self.avg_pool[0](drug_fea)
        food_fea = self.avg_pool[1](food_fea)
        DF_Feature = torch.cat((drug_fea, food_fea, temp), 2)
        DF_Feature_one,_ = self.DF_Multihead(DF_Feature, DF_Feature, DF_Feature)
        DF_Feature_one = self.adap_pool(DF_Feature_one.permute(0,2,1)).permute(0, 2, 1)

        ###############################Structural Feature Extractor######################################
        DF_Feature_two, drug_attn_list = self.model_two(dataset_batch)
        ##################################       Classifier        ######################################
        DF_Feature = self.Fusion(DF_Feature_one, DF_Feature_two, drug_IDs+food_IDs)

        DF_Feature = DF_Feature_one+DF_Feature_two
        DF_Feature = DF_Feature.view(DF_Feature.shape[0], -1) 

        for i in range(2):
            DF_Feature = self.drop_out[i](self.act(self.Batch[i](self.DFI_feature[i](DF_Feature))))
        DF_Feature = self.DFI_Pre(DF_Feature )

        return DF_Feature
        #return DF_Feature,self.class_imbalance2(temp, labels.to(device))