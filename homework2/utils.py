# -*- coding: utf-8 -*-
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
import os

import matplotlib
# There is no display in a remote serv which means that matplotlib will give an error unless we change the display 
#to Agg backend and save the figure.
if os.environ.get('DISPLAY','') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

def train(model, train_loader, dev_loader, optim="Adam", lr=0.01, weight_decay=0., epochs=5, cuda=False):
    """
    This trains the perceptron over a certain number of epoch and records the accuracy in Train and Dev sets along each epoch.
    :param model: Pytorch model to be trained.
    :param X: numpy array with size DxN where D is the number of training examples and N is the number of features.
    :param Y: numpy array with size D containing the correct labels for the training set
    :param devX  (optional): same as X but for the dev set.
    :param devY  (optional): same as Y but for the dev set.
    :param optim (optional): Name of the optimizer used during training.
    :param lr      (optional): learning rate to be used.
    :param weight_decay (optional): Regularization constant.
    :param epochs (optional): number of epochs to run
    :param cuda (optional): Flag to train the model using GPU device.

    Note: This function will print a loading bar ate the terminal for each epoch.
    """
    if cuda:
        model.cuda()
    optim = getattr(torch.optim, optim)
    optimizer = optim(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = nn.CrossEntropyLoss()
    train_accuracy = [evaluate(model, train_loader)]
    dev_accuracy = [evaluate(model, dev_loader)]
    print ("Dev Accuracy: {0:.4f}".format(dev_accuracy[0]))
    print ("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            # Clear gradients
            model.zero_grad()
            # Prepare inputs
            inputs = Variable(inputs.float()).cuda() if cuda else Variable(inputs.float())
            labels = Variable(labels.long()).cuda() if cuda else Variable(labels.long())
            # compute probabilities
            probs = model(inputs)
            loss = loss_func(probs, labels)
            loss.backward()
            optimizer.step()
            # sum losses across each epoch.
            total_loss += loss.item()
        train_accuracy.append(evaluate(model, train_loader))
        dev_accuracy.append(evaluate(model, dev_loader))
        print ("Loss: {0:.4f} Train Accuracy: {1:.3f} Dev Accuracy: {2:.3f} ".format(
            total_loss / len(train_loader), train_accuracy[-1], dev_accuracy[-1])
        )
    return model, train_accuracy, dev_accuracy

def evaluate(model, loader):
    """
    Evaluates the error in a given set of examples.
    :param model: Pytorch model to evaluate.
    :param loader: Pytorch data loader.
    """
    total = 0
    model.eval()
    correct_predictions = 0
    for i, (inputs, labels) in enumerate(loader):
        inputs = Variable(inputs.float()).cuda() if next(model.parameters()).is_cuda else Variable(inputs.float())
        labels = Variable(labels).cuda() if next(model.parameters()).is_cuda else Variable(labels)
        predictions = model(inputs).argmax(dim=1)
        correct_predictions += int((labels.cpu() == predictions.cpu()).sum())
        total += labels.shape[0]
    return correct_predictions/int(total)

def plot_train(train_accuracy, dev_accuracy, filename):
    """
    Function to Plot the accuracy of the Training set and Dev set per epoch.
    :param train_accuracy: list containing the accuracies of the train set.
    :param dev_accuracy: list containing the accuracies of the dev set.
    """
    x_axis = [epoch+1 for epoch in range(len(train_accuracy))]
    plt.plot(x_axis, train_accuracy, '-g', linewidth=1, label='Train')
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    plt.plot(x_axis, dev_accuracy, 'b-', linewidth=1, label='Dev')
    plt.legend()
    plt.savefig("plots/"+filename)
    plt.gcf().clear()


