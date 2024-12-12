import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, hidden_dim=[1000, 2048, 512]):
        super(MLP, self).__init__()
        self.input_dim = hidden_dim[0]
        self.hidden_dim = hidden_dim
        orderedDict = OrderedDict()
        for i in range(len(hidden_dim) - 1):
            index = i + 1
            orderedDict['linear' + str(index)] = nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1])
            orderedDict['bn' + str(index)] = nn.BatchNorm1d(self.hidden_dim[i + 1])
            orderedDict['act' + str(index)] = nn.ReLU()
        self.mlp = nn.Sequential(orderedDict)

    def forward(self, x):
        x = x.float()
        return self.mlp(x)

class Hash_Net(nn.Module):
    def __init__(self, image_hidden_dim, txt_hidden_dim, bit, num_classes):
        super(Hash_Net, self).__init__()
        self.bit = bit
        self.img_hidden_dim = image_hidden_dim
        self.txt_hidden_dim = txt_hidden_dim
        self.fusion_dim = image_hidden_dim[-1]
        self.imageMLP = MLP(hidden_dim=self.img_hidden_dim)
        self.textMLP = MLP(hidden_dim=self.txt_hidden_dim)
        self.hash = nn.Sequential(
            nn.Linear(self.fusion_dim, self.bit),
            nn.BatchNorm1d(self.bit),
            nn.Tanh())
        self.centroids = nn.Parameter(torch.randn(num_classes, self.bit)).to(dtype=torch.float32)

    def forward(self, x, y):
        imageH = self.imageMLP(x)
        textH = self.textMLP(y)
        fusionH = imageH + textH
        hash_code = self.hash(fusionH)
        return hash_code
