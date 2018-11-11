# -*- coding: utf-8 -*-
from sklearn.externals import joblib
import torch.utils.data as utils
from utils import *
import numpy as np
import argparse
import torch


class LogisticRegression(nn.Module):
    """ LogisticRegression: Pytorch implementation of a Logistic Regression module. """
    
    def __init__(self, input_size, n_classes):
        """
        :param input_size: Input size expected.
        :param n_classes: Number of classes.
        """
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, n_classes)
        self.linear.weight.data = torch.zeros(self.linear.weight.shape)

    def forward(self, x):
        """
        Returns the a score over the set of classes.
        :param x: Input that we want to classify.
        """
        return self.linear(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Epochs to run.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate to be used.")
    parser.add_argument("--optim", type=str, default="Adam", help="Optimizer function name.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay constant to be used (Regularization constant).")
    parser.add_argument("--batch_size", type=int, default=256, help="Minibatch size to use during train.")
    parser.add_argument("--cuda", type=bool, default=True, help="Flag to run the training in a GPU device.")
    args = parser.parse_args()

    # Load data.
    train_x, train_y = joblib.load("data/train.pkl")
    dev_x, dev_y = joblib.load("data/dev.pkl")

    train_data = utils.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    dev_data = utils.TensorDataset(torch.from_numpy(dev_x), torch.from_numpy(dev_y))
    train_loader = utils.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    dev_loader = utils.DataLoader(dev_data, batch_size=args.batch_size, shuffle=True)
    
    # initialize model
    lr = LogisticRegression(train_x.shape[1], np.unique(train_y).shape[0])

    # start training
    lr, train_accuracy, dev_accuracy = train(
        lr, train_loader, dev_loader, 
        weight_decay=args.weight_decay, optim=args.optim, lr=args.lr, epochs=args.epochs, cuda=args.cuda
    )

    # plot accuracies during training.
    plot_train(train_accuracy, dev_accuracy, "lr-accuracy.png")
    
if __name__ == '__main__':
    main()




    



