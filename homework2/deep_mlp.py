# -*- coding: utf-8 -*-
from sklearn.externals import joblib
import torch.utils.data as utils
import torch.nn.functional as F
from utils import *
import numpy as np
import argparse
import torch


class ListModule(nn.Module):
    """ ListModule: Auxiliar module that works as a list of modules.
                    This class is necessary to store modules inside lists. """
    def __init__(self, *args):
        """
        :param *args: Receives a set of modules. A list containing all the modules is prefered.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        :param idx: Indexe of the item we want to access..
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """ Allows iterations over the modules. """
        return iter(self._modules.values())

    def __len__(self):
        """ Returns the number of modules stored inside ListModule class. """
        return len(self._modules)

class DeepFeedForwardNN(nn.Module):
    """ DeepFeedForwardNN: Pytorch implementation of a feed forward neural network with several possible layers.. """
    
    def __init__(self, input_size, n_classes, layers=1, dropout=0., activation_func="Sigmoid"):
        """
        :param input_size: Input size expected..
        :param n_classes: Number of classes.
        :param layers: Number of hidden layers.
        :param dropout: dropout to be used between hidden layers.
        :param activation_func: Name of the activation function to be used 
                                (see torch.nn documentation to see available activations).
        """
        super(DeepFeedForwardNN, self).__init__()
        self.dropout_value = dropout
        self.dropout = nn.Dropout(p=dropout)
        
        self.hidden_linear_layers = []
        for i in range(layers):
            self.hidden_linear_layers.append(nn.Linear(input_size, input_size))
        self.hidden_linear_layers = ListModule(*self.hidden_linear_layers)
        
        self.output_linear = nn.Linear(input_size, n_classes)
        activation = getattr(nn, activation_func)
        self.activation = activation()

    def forward(self, x):
        """
        Returns the softmax distribution of x over the set of classes.
        :param x: Input that we want to classify.
        """
        linear0 = self.hidden_linear_layers[0]
        layeri_out= self.activation(linear0(x))
        for i in range(1, len(self.hidden_linear_layers)):
            lineari = self.hidden_linear_layers[i]
            layeri_out = self.dropout(self.activation(lineari(layeri_out)))
        return self.dropout(self.output_linear(layeri_out))


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
    args = parser.parse_args()

    # Load data.
    train_x, train_y = joblib.load("data/train.pkl")
    dev_x, dev_y = joblib.load("data/dev.pkl")

    train_data = utils.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    dev_data = utils.TensorDataset(torch.from_numpy(dev_x), torch.from_numpy(dev_y))
    train_loader = utils.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    dev_loader = utils.DataLoader(dev_data, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    dff_nn = DeepFeedForwardNN(
        train_x.shape[1], np.unique(train_y).shape[0], 
        layers=args.layers, dropout=args.dropout, activation_func=args.activation
    )

    # Start training.
    dff_nn, train_accuracy, dev_accuracy = train(
        dff_nn, train_loader, dev_loader, optim=args.optim, lr=args.lr, epochs=args.epochs, cuda=args.cuda
    )
    
    # Plot accuracies during training.
    plot_train(train_accuracy, dev_accuracy, "deep-feed-forward-accuracy.png")
    
if __name__ == '__main__':
    main()