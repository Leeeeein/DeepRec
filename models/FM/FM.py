import torch
import torch.nn as nn
import os
from dataset.dataloader import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import argparse
import matplotlib.pyplot as plt

class FM(nn.Module):

    def __init__(self, field_dim,
                 embed_dim):
        super(FM, self).__init__()
        self.w = nn.Linear(field_dim, 1, bias=True)
        self.v = nn.Parameter(torch.FloatTensor(field_dim, embed_dim), requires_grad=True)
        nn.init.xavier_normal_(self.v)

    def forward(self, input):
        """
        input shape: (num_item, field_dim)
        """
        linear_comb = self.w(input)
        square_of_sum = torch.mm(input, self.v)
        square_of_sum = torch.pow(square_of_sum, 2)
        first_term = torch.sum(square_of_sum, dim=1)

        square_of_v = torch.pow(self.v, 2)
        square_of_x = torch.pow(input, 2)
        sum_of_square = torch.mm(square_of_x, square_of_v)
        second_term = torch.sum(sum_of_square, dim=1)
        output = first_term - second_term
        p = torch.sigmoid(linear_comb.squeeze(1) + output)
        return p.view(-1, 1)


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

    model = FM(train_inputs.shape[1], embed_dim)
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
    parser.add_argument('--embed_dim', default=1024)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
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