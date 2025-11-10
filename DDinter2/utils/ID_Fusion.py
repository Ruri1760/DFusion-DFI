import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

class ID_Fusion(nn.Module):
    def __init__(self, hidden1, hidden2):
        super(ID_Fusion, self).__init__()
        self.W_q = nn.Linear(hidden1, hidden2)
        self.W_k = nn.Linear(hidden2, hidden2)
        self.W_v = nn.Linear(hidden2, hidden2)
        self.drop = nn.Dropout(p=0.1)
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(-1)

    def forward(self, x, y, z): 
        q = self.W_q(x)
        k = self.W_k(y)
        v = self.W_v(z)
        weight = torch.matmul(q, k.permute(0, 2, 1))
        scale = weight.size(-1) ** -0.5
        weights = self.softmax(weight * scale)
        ys = torch.matmul(self.drop(weights), v)+x+y
        return ys

