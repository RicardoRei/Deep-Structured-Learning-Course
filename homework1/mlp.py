# -*- coding: utf-8 -*-
from sklearn.externals import joblib
from classifier import Classifier
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class MLP(Classifier):
    """ Multi Class Perceptron """

    def __init__(self, input_size, n_classes, hidden_layers=1, lr=0.01):
        """
        Initializes all the parameters for the Multi-Layer Perceptron.
        :param input_size: Number of features.
        :param n_classes: Number of classes to classify the inputs.
        :param hidden_layers (Default=1): Number of hidden layers to consider.
        :param lr: Learning rate to be used in the gradient descent algorithm.
        """
        self.lr = lr
        self.hidden_layers = hidden_layers
        # This initialization is not the best one but by scalling it to small values arround zero achieves satisfactory results.
        self.W = [np.random.rand(input_size, input_size+1)*1e-2 for i in range(hidden_layers)]
        self.W.append(np.random.rand(n_classes, input_size+1)*1e-2)

    def sigmoid(self, x):
        """ Compute element-wise sigmoid. """
        return (1 / (1 + np.exp(-x)))

    def sigmoid_derivative(self, x):
        """ Compute the element-wise derivative of the sigmoid function. """
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def softmax(self, x):
        """ Compute softmax values for each sets of scores in x."""
        return np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x)), axis=0)

    def forward(self, x):
        # We compute the forward for the first layer and the hidden layers together.
        Ai = x
        # We need to store the linear tranformations and activations for the backward propagation
        self.Zi_layer = []
        self.Ai_layer = [Ai]
        for i in range(0, self.hidden_layers):
            Zi = np.dot(self.W[i], np.append(Ai, [1]))
            Ai = self.sigmoid(Zi)
            self.Zi_layer.append(Zi)
            self.Ai_layer.append(Ai)
        # The last layer has a different activation function and for that reason we compute this layer independently.
        Zi = np.dot(self.W[self.hidden_layers], np.append(Ai, [1]))
        self.Zi_layer.append(Zi)
        return self.softmax(Zi)

    def backward(self, y_pred, target):
        """
        This function will run the backward algorithm and update all the network paramenters. This functions assumes 
        a Cross-Entropy Loss and sigmoid activation functions.
        :param y_pred: Numpy array containing the probability of each class after the forward algorithm.
        :param target: Numpy array containing the target distribution (one-hot vector of the true label).
         """
        gradients = []
        # Last layer gradients.
        dZi = y_pred - target
        
        dWi = np.outer(dZi, self.Ai_layer[-1])
        gradients.append((dWi, dZi))

        # Input and Hidden layers gradients.
        for i in range(self.hidden_layers, 0, -1):
            dZi = np.dot(self.W[i][:, :-1].T, dZi)*self.sigmoid_derivative(self.Zi_layer[i-1])
            dWi = np.outer(dZi, self.Ai_layer[i-1])
            gradients.append((dWi, dZi))
        
        # Update paramenters with gradients.
        for i in range(len(gradients)):
            dWi, dZi = gradients.pop()
            # W update: W = W - lr*dWi
            self.W[i][:, :-1] = self.W[i][:, :-1] - self.lr*dWi
            # Bias update: Bias = Bias
            self.W[i][:, -1] = self.W[i][:, -1] - self.lr*dZi

    def predict(self, x):
        """
        This function will run the forward algorithm and select the argmax of the final predictions.
        :param x: numpy array with size 1xN where N = number of features.
        """
        return np.argmax(self.forward(x))

    def update_weights(self, x, y):
        """
        Function that will take an input example and the true prediction and will update the model parameters.
        :param x: Array of size N where N its the number of features that the model takes as input.
        :param y: The int corresponding to the correct label.
        """
        predictions = self.forward(x)
        self.backward(predictions, np.eye(1, predictions.shape[0], y)[0])


def main():
    train_x, train_y = joblib.load("data/train.pkl")
    dev_x, dev_y = joblib.load("data/dev.pkl")
    test_x, test_y = joblib.load("data/test.pkl")
    mlp = MLP(train_x.shape[1], np.unique(train_y).shape[0], 2)
    train_accuracy, dev_accuracy = mlp.train(train_x, train_y, dev_x, dev_y, epochs=30)
    print ("Train Accuracy: {}".format(mlp.evaluate(train_x, train_y)))
    print ("Dev Accuracy: {}".format(mlp.evaluate(dev_x, dev_y)))
    print ("Test Accuracy: {}".format(mlp.evaluate(test_x, test_y)))
    mlp.plot_train(train_accuracy, dev_accuracy, "mlp-accuracy.png")

if __name__ == '__main__':
    main()
        


        