import torch
import torch.nn as nn
import os
from dataset.dataloader import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import argparse
import matplotlib.pyplot as plt

class WideDeep(nn.Module):
    def __init__(self, input_dim,
                 embed_dim,
                 layers_dim,
                 dropout,
                 output_layer = False):
        super(WideDeep, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.layers_dim = layers_dim

        self.act = nn.Sigmoid()
        self.linear = nn.Linear(input_dim, 1)

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
        self.output = nn.Linear(layers_dim[-1], 1)

    def forward(self, input):
        output_linear = self.linear(input)
        output_mlp = self.mlp(input)
        deep = self.output(output_mlp)

        return self.act(deep + output_linear)

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
    # (self, input_dim,
    #  embed_dim,
    #  layers_dim,
    #  dropout,
    #  output_layer = False)
    model = WideDeep(X_train.shape[1], 64, [128, 64, 16], 0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    loss_list = list()
    auc_list = []
    for epoch in range(epochs):
         total_loss = 0
         for it, (x, y) in enumerate(train_loader):
             pred = model(x)
             # print(pred.shape)
             # print(y.shape)
             loss = criterion(pred, y)
             model.zero_grad()
             loss.backward()
             optimizer.step()
             total_loss += loss.item()
             loss_list.append(loss.item())
             #if it % 30 == 0:
             print(f'    epochs:[{epoch}], iter:[{it}], average loss:[{total_loss}]')
             total_loss = 0
             predict = model(test_inputs)
             target = test_targets
             auc = roc_auc_score(target.detach().numpy(), predict.detach().numpy())
             auc_list.append(auc)
         print(f'** auc.[{auc}]', auc)
    plt.plot(range(len(auc_list)), auc_list)
    plt.show()

if __name__ == '__main__':
    #print(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', default=1024)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--test_ratio', type=float, default=0.5)
    args = parser.parse_args()
    trainer(args.embed_dim,
            args.learning_rate,
            args.weight_decay,
            args.epochs,
            args.batch_size,
            args.train_ratio,
            args.test_ratio)

