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

    def __init__(self, clayers_num, dlayers_num, input_dim, output_dim, dropout, output_layer = True):
        super(DeepCrossNetwork, self).__init__()
        self.cln = clayers_num
        self.dln = dlayers_num
        self.input_dim = input_dim
        self.w = nn.ModuleList([torch.nn.Linear(self.input_dim, 1) for _ in range(self.cln)])
        layers = []
        for embed_dim in range(self.dln):
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
    # __init__(self, clayers_num, dlayers_num, input_dim, output_dim, dropout, dlayers_dims, output_layer = True):
    model = DeepCrossNetwork(3, 3, train_inputs.shape[1], embed_dim, 0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    loss_list = list()
    for epoch in range(epochs):
        total_loss = 0
        for it, (x, y) in enumerate(train_loader):
            pred = model(x)
            loss = criterion(pred, y)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loss_list.append(loss.item())
            #if it % 30 == 0:
            print(f'    epochs:[{epoch}], iter:[{it}], average loss:[{total_loss}]')
            total_loss = 0
        predict = model(val_inputs)
        target = val_targets
        auc = roc_auc_score(target.detach().numpy(), predict.detach().numpy())
        print(f'epochs:[{epoch}], auc:[{auc}]')
    plt.plot(range(len(loss_list)), loss_list)
    plt.show()

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

