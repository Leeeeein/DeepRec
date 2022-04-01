import torch
import torch.nn as nn
import os
from dataset.dataloader import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import argparse
import matplotlib.pyplot as plt
import tqdm

class DeepCrossNetwork(nn.Module):

    def __init__(self, clayers_num, dlayers_num, input_dim, output_dim, dropout, dlayers_dims, output_layer = True):
        super(DeepCrossNetwork, self).__init__()
        self.cln = clayers_num
        self.dln = dlayers_num
        self.input_dim = input_dim
        self.w = nn.ModuleList([torch.nn.Linear(self.input_dim, 1) for _ in range(self.cln)])

        layers = ()
        for embed_dim in dlayers_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self.output = nn.Linear(output_dim * 2, 1)

    def forward(self, input):
        output_mlp = self.mlp(input)
        x0 = input
        output_cross = input
        for i in range(self.cln):
            xw = self.w[i](output_cross)
            output_cross =  x0 * xw + input
        output_concat = torch.concat([output_mlp, output_cross], dim=1)
        output = self.output(output_concat)
        return nn.Sigmoid(output)


