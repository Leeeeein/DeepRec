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
        self.v_fields = nn.ParameterList([torch.nn.Parameter(
            torch.FloatTensor(field_dim, embed_dim), requires_grad=True) for _ in range(fields_num)])
        for v_f in self.v_fields:
            nn.init.xavier_normal_(v_f)

    def forward(self, input):
        linear_comb = self.w(input)

        vs = [torch.mm(input, v_f) for v_f in self.v_fields]
        for i in range(self.fields_num):
            for j in range(i+1, self.fields_num):
                product = vs[j][:,i], vs[i][:,j]
        return product

def get_data():
    X_train, Y_train, X_test, Y_test, X_val, Y_val = DataLoading(args.train_ratio, args.test_ratio)
    return X_train, Y_train, X_test, Y_test, X_val, Y_val

def trainer(embed_dim, learning_rate, weight_decay, epochs, batch_size, train_ratio, test_ratio):
    X_train, Y_train, X_test, Y_test, X_val, Y_val = get_data()
    train_inputs, train_targets = torch.FloatTensor(X_train), torch.FloatTensor(Y_train)
    test_inputs, test_targets = torch.FloatTensor(X_test), torch.FloatTensor(Y_test)
    val_inputs, val_targets = torch.FloatTensor(X_val), torch.FloatTensor(Y_val)

    train_dataset = TensorDataset(train_inputs, train_targets)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = FieldAwareFM(train_inputs.shape[1], embed_dim, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    loss_list = list()
    for epoch in range(epochs):
        total_loss = 0
        for it, (x, y) in enumerate(train_loader):
            u = model(x)
            print(u)

if __name__ == '__main__':
    #print(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--test_ratio', type=float, default=0.5)
    args = parser.parse_args()
    trainer(args.embed_dim,
            args.learning_rate,
            args.weight_decay,
            args.epochs,
            args.batch_size,
            args.train_ratio,
            args.test_ratio)