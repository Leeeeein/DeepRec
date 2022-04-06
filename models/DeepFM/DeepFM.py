import torch
import torch.nn as nn
import os
from dataset.dataloader import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import argparse
import matplotlib.pyplot as plt
import tqdm

class DeepFM(nn.Module):
    def __init__(self, clayers_num, dlayers_num, input_dim, layers_dim, dropout, output_layer = False):
        super(DeepFM, self).__init__()
        self.w = nn.Linear(field_dim, 1, bias=True)
        self.v = nn.Parameter(torch.FloatTensor(field_dim, embed_dim), requires_grad=True)
        nn.init.xavier_normal_(self.v)

        layers = []
        for embed_dim in layers_dim:
            layers.append(torch.nn.Linear(self.input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            self.input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(input_dim + layers_dim[-1], 1)

    def forward(self, input):
        linear_comb = self.w(input)
        second_order = 0
        for i in range(self.feats_dim):
            for j in range(i+1, self.feats_dim):
                inner_product = torch.dot(self.vs[self.fields_dict[j], i, :], self.vs[self.fields_dict[i], j, :])
                second_order += torch.sum(inner_product * input[:,i] * input[:,j])
        return torch.sigmod(linear_comb + second_order)
