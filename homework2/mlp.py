# -*- coding: utf-8 -*-
from sklearn.externals import joblib
import torch.utils.data as utils
from utils import *
import numpy as np
import argparse
import torch

class FeedForwardNN(nn.Module):
    """ FeedForwardNN: Pytorch implementation of a single layer feed forward neural network. """
    
    def __init__(self, input_size, n_classes, hidden_size, dropout=0, activation_func="Sigmoid"):
        """
        :param input_size: Input size expected..
        :param n_classes: Number of classes.
        :param hidden_size: Size of the hidden layers.
        :param dropout: dropout to be used after the hidden layer activation.
        :param activation_func: Name of the activation function to be used 
                                (see torch.nn documentation to see available activations).
        """
        super(FeedForwardNN, self).__init__()
        self.dropout_value = dropout
        self.dropout = nn.Dropout(p=dropout)
        self.l0_linear = nn.Linear(input_size, hidden_size)
        self.l1_linear = nn.Linear(hidden_size, n_classes)
        activation = getattr(nn, activation_func)
        self.activation = activation()

    def forward(self, x):
        """
        Returns the softmax distribution of x over the set of classes.
        :param x: Input that we want to classify.
        """
        layer0_out = self.activation(self.l0_linear(x))
        return self.dropout(self.l1_linear(layer0_out))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout", type=float, default=0., help="Dropout value to apply during train.")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs to run.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate to be used.")
    parser.add_argument("--activation", type=str, default="Sigmoid", help="Activation function name.")
    parser.add_argument("--optim", type=str, default="Adam", help="Optimizer function name.")
    parser.add_argument("--layers", type=int, default=2, help="Number of hidden layers.")
    parser.add_argument("--batch_size", type=int, default=256, help="Minibatch size to use during train.")
    parser.add_argument("--cuda", type=bool, default=True, help="Flag to run the training in a GPU device.")
    parser.add_argument("--hidden_size", type=int, default=256, help="Number of hidden units.")
    args = parser.parse_args()

    # Load data.
    train_x, train_y = joblib.load("data/train.pkl")
    dev_x, dev_y = joblib.load("data/dev.pkl")
    test_x, test_y = joblib.load("data/test.pkl")
    train_data = utils.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    dev_data = utils.TensorDataset(torch.from_numpy(dev_x), torch.from_numpy(dev_y))
    test_data = utils.TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    train_loader = utils.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    dev_loader = utils.DataLoader(dev_data, batch_size=args.batch_size, shuffle=True)
    test_loader = utils.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    # Initialize model.
    ff_nn = FeedForwardNN(
        train_x.shape[1], np.unique(train_y).shape[0], args.hidden_size, 
        dropout=args.dropout, activation_func=args.activation
    )
    
    # Start training.
    ff_nn, train_accuracy, dev_accuracy = train(
        ff_nn, train_loader, dev_loader, optim=args.optim, lr=args.lr, epochs=args.epochs, cuda=args.cuda
    )

    #print (evaluate(ff_nn, test_loader))
    # plot accuracies during training.
    plot_train(train_accuracy, dev_accuracy, "mlp-accuracy.png")
    
if __name__ == '__main__':
    main()




    



