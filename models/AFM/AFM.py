import torch
import torch.nn as nn
from dataset.dataloader import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
import tqdm

class AttentionalFM(nn.Module):
    def __init__(self,
                 fields_dim,
                 embed_dim,
                 attn_dim):
        super(AttentionalFM, self).__init__()
        self.v = nn.Parameter(torch.FloatTensor(fields_dim, embed_dim))
        self.w = nn.Linear(embed_dim, attn_dim, bias=True)
        self.h = nn.Linear(attn_dim, 1)
        self.p = nn.Linear(embed_dim, 1)

    def forward(self, input):
        #input size: (batch_size, field_dim, field_value)
        vx = self.v * input
        #vx size: (batch_size, field_dim, emebd_dim)
        fields_num = input.shape[1]
        row, col = [], []
        for r in range(fields_num-1):
            for c in range(r+1, fields_num):
                row.append(r)
                col.append(c)
        vp, vq = vx[:, row], vx[:, col]
        element_wise = vp * vq # (batch, sum(ij), embed_dim)
        _a = self.w(element_wise) # (batch, sum(ij), attn_dim)
        _a = F.relu(_a) # (batch, sum(ij), attn_dim)
        _a = self.h(_a) # (batch, sum(ij), 1)
        a = F.softmax(_a, dim=1) # (batch, sum(ij), 1)
        a = a * element_wise
        a = self.p(a)

class AttentionalFM(nn.Module):
    def __init__(self, fields_dim,
                 feats_dim,
                 attn_dim,
                 training,
                 dropout,
                 fields_dict):
        super(AttentionalFM, self).__init__()
        self.w = nn.Linear(fields_dim, attn_dim, bias=True)
        self.h = nn.Linear(attn_dim, 1)

        self.fields_dict = fields_dict
        self.fields_dim = fields_dim
        self.feats_dim = feats_dim
        self.vs = nn.Parameter(torch.FloatTensor(fields_dim, feats_dim, attn_dim))
        self.w = nn.Linear(attn_dim, attn_dim, bias=True)
        nn.init.xavier_normal_(self.vs)

        self.p = nn.Linear(attn_dim, 1)
        self.dropout = dropout
        self.training = training


    def __forward_fields(self, input):
        pass

    def __forward_dense(self, input):
        _row, _col = [], []
        row, col = [], []
        fields_num = input.shape[1]
        for i in range(fields_num):
            for j in range(fields_num):
                _row.append(i)
                _col.append(j)
        for i in range(len(_row)):
            row.append(self.fields_dict[_row[i]])
            col.append(self.fields_dict[_col[i]])
        p, q = self.vs[row, :], self.vs[col, :]
        inner_product = p * q
        print(inner_product.shape)
        _a = self.h(F.relu(self.w(inner_product)))
        a = F.softmax(_a, dim=1)
        a = F.dropout(a, self.dropout, training=self.training)
        a = torch.sum(a * inner_product, dim=1)
        a = F.dropout(a, self.dropout, training=self.training)
        return self.p(a)
    def forward(self, input):
        return self.__forward_dense(input)

def get_data(train_ratio, test_ratio):
    X_train, Y_train, X_test, Y_test, X_val, Y_val, dic = DataLoading(train_ratio, test_ratio, True)
    return X_train, Y_train, X_test, Y_test, X_val, Y_val, dic

def trainer(attn_dim, learning_rate, weight_decay, epochs, batch_size, train_ratio, test_ratio):
    X_train, Y_train, X_test, Y_test, X_val, Y_val, dic = get_data(train_ratio, test_ratio)
    train_inputs, train_targets = torch.FloatTensor(X_train), torch.FloatTensor(Y_train)
    test_inputs, test_targets = torch.FloatTensor(X_test), torch.FloatTensor(Y_test)
    val_inputs, val_targets = torch.FloatTensor(X_val), torch.FloatTensor(Y_val)

    train_dataset = TensorDataset(train_inputs, train_targets)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

#fields_dim,
#                 feats_dim,
#                 attn_dim,
#                 training,
#                 dropout,
#                 fields_dict

    model = AttentionalFM(39, train_inputs.shape[1], attn_dim,  True, 0.5, dic)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    loss_list = list()
    for epoch in tqdm.tqdm(range(epochs)):
        total_loss = 0
        for it, (x, y) in enumerate(train_loader):
            u = model(x)

if __name__ == '__main__':
    #print(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--attn_dim', default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--test_ratio', type=float, default=0.5)
    args = parser.parse_args()
    trainer(args.attn_dim,
            args.learning_rate,
            args.weight_decay,
            args.epochs,
            args.batch_size,
            args.train_ratio,
            args.test_ratio)