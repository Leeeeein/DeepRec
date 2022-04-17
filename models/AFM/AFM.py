import torch
import torch.nn as nn
from dataset.dataloader import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F

class AttentionalFM(nn.Module):
    def __init__(self, field_dim,
                 embed_dim,
                 attn_dim,
                 training,
                 dropout):
        super(self, AttentionalFM).__init__()
        self.w = nn.Linear(field_dim, attn_dim, bias=True)
        self.h = nn.Linear(attn_dim, 1)
        self.v = nn.Parameter(torch.FloatTensor(field_dim, embed_dim), requires_grad=True)
        self.p = nn.Linear(embed_dim, 1)
        self.dropout = dropout
        self.training = training
        nn.init.xavier_normal_(self.v)

    def __forward_fields(self):
        pass

    def __forward_dense(self, input):
        """
        input: shape is (batch_size, fields_size, field_dim)
        """
        fields_num = input.shape[1]
        row = [], col = []
        for i in range(fields_num-1):
            for j in range(i+1, fields_num):
                row.append(i)
                col.append(j)
        p, q = input[:, row], input[:, col]
        inner_product = p * q
        _a = self.h(F.relu(self.w(inner_product)))
        a = F.softmax(_a, dim=1)
        a = F.dropout(a, self.dropout, training=self.training)
        a = torch.sum(a * inner_product, dim=1)
        a = F.dropout(a, self.dropout, training=self.training)
        return self.p(a)






