import torch
import torch.nn as nn
import os
from dataset.dataloader import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import argparse
import matplotlib.pyplot as plt

class FieldAwareFM(nn.Module):

    def __init__(self, feats_dim, embed_dim, fields_num):
        #fields_dict take a dict as input, every item of field_dict is {feature_idx, field_idx}
        super(FieldAwareFM, self).__init__()
        self.fields_num = fields_num
        self.fields_dim = feats_dim
        self.embed_dim = embed_dim
        self.w = nn.Linear(feats_dim, 1, bias=True)
        # field_num indicates the number of fields while fields_dim indicates features length including dense features and sparse features
        self.vs = nn.Parameter(torch.FloatTensor(fields_num, feats_dim, embed_dim))
        nn.init.xavier_normal_(self.vs)

    def forward(self, input, fields_dict):
        linear_comb = self.w(input)
        second_order = 0
        for i in range(self.feats_dim):
            for j in range(i+1, self.feats_dim):
                inner_product = torch.dot(self.vs[fields_dict[j], i, :], self.vs[fields_dict[i], j, :])
                second_order += torch.sum(inner_product * input[:,i] * input[:,j])
        return torch.sigmod(linear_comb + second_order)

def get_data(train_ratio, test_ratio):
    X_train, Y_train, X_test, Y_test, X_val, Y_val = DataLoading(train_ratio, test_ratio)
    return X_train, Y_train, X_test, Y_test, X_val, Y_val

def trainer(embed_dim, learning_rate, weight_decay, epochs, batch_size, train_ratio, test_ratio):
    X_train, Y_train, X_test, Y_test, X_val, Y_val = get_data(train_ratio, test_ratio)
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