import torch
import torch.nn as nn
import os
from dataset.dataloader import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import argparse
import matplotlib.pyplot as plt

class FieldAwareFM(nn.Module):

    def __init__(self, field_dim, embed_dim, fields_num):
        super(FieldAwareFM, self).__init__()
        self.fields_num = fields_num
        self.w = nn.Linear(field_dim, 1, bias=True)
        self.v_fields = nn.ModuleList([torch.nn.Parameter(
            torch.FloatTensor(field_dim, embed_dim), requires_grad=True) for _ in range(fields_num)])
        for v_f in self.v:
            nn.init.xavier_normal_(v_f)

    def forward(self, input):
        linear_comb = self.w(input)

        vs = [torch.mm(input, v_f) for v_f in self.v_fields]
        for i in range(self.fields_num):
            for j in range(self.fields_num):
                product = torch.mm(vs[j][:,i], vs[i][:,j])