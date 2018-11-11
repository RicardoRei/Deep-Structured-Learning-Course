# -*- coding: utf-8 -*-
from torch.autograd import Variable
import torch.utils.data as utils
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import string
import torch
import os

import matplotlib
# There is no display in a remote serv which means that matplotlib will give an error unless we change the display 
#to Agg backend and save the figure.
if os.environ.get('DISPLAY','') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

def train(model, train_loader, dev_loader, optim="Adam", lr=0.001, weight_decay=0., epochs=5, cuda=True):
    """
    This trains the perceptron over a certain number of epoch and records the accuracy in Train and Dev sets along each epoch.
    :param model: Pytorch model to be trained.
    :param train_loader: Pytorch Dataloader used to fetch the training data.
    :param dev_loader: Pytorch Dataloader used to fetch the dev data.
    :param optim (optional): Name of the optimizer used during training.
    :param lr (optional): learning rate to be used.
    :param weight_decay (optional): Regularization constant.
    :param epochs (optional): number of epochs to run
    :param cuda: Flag to train the model using a GPU.
    Note: This function will print a loading bar ate the terminal for each epoch.
    """
    model.cuda() if cuda else None
    optim = getattr(torch.optim, optim)
    optimizer = optim(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = nn.CrossEntropyLoss()
    train_accuracy = [evaluate(model, train_loader)]
    dev_accuracy = [evaluate(model, dev_loader)]
    print ("Train Accuracy: {0:.6f} Dev Accuracy: {1:.6f}".format(train_accuracy[0], dev_accuracy[0]))
    print ("Starting training...")
    for epoch in range(epochs):
        total_loss = 0
        input_count = 0
        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            model.train()
            # Clear gradients
            optimizer.zero_grad()
            # Prepare input and target.
            inputs = Variable(inputs.cuda()) if cuda else Variable(inputs)
            target = Variable(labels).cuda() if cuda else Variable(labels)
            # Compute scores
            scores = model(inputs)
            # Target and scores size assert for the rnn model
            target = target.squeeze(0) if target.shape[0] != scores.shape[0] else target
            loss = loss_func(scores, target)
            loss.backward()
            optimizer.step()
            # sum losses across each epoch.
            total_loss += loss.item() 
            # count number of elements for epoch.
        # evaluate the model over training and dev sets.
        train_accuracy.append(evaluate(model, train_loader))
        dev_accuracy.append(evaluate(model, dev_loader))
        print ("Loss: {0:.6f} Train Accuracy: {1:.6f} Dev Accuracy: {2:.6f}".format(
            total_loss/len(train_loader), train_accuracy[-1], dev_accuracy[-1])
        )
    return model, train_accuracy, dev_accuracy

def evaluate(model, dataloader):
    """
    Evaluates the error in a given set of examples.
    :param model: Pytorch model to evaluate.
    :param dataloader: Pytorch Dataloader used to fetch the data to be evaluated.
    """
    model.eval()
    correct_predictions = 0
    total = 0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = Variable(inputs.cuda()) if next(model.parameters()).is_cuda else Variable(inputs)
        y_pred = model(inputs).argmax(dim=1)
        correct_predictions += int((labels.cpu() == y_pred.cpu()).sum())
        total += labels.shape[0] if labels.shape[0] > 1 else labels.shape[1]
    return correct_predictions/int(total)

def plot_train(train_accuracy, dev_accuracy, filename):
    """
    Function to Plot the accuracy of the Training set and Dev set per epoch.
    :param train_accuracy: List containing the accuracies of the train set.
    :param dev_accuracy: List containing the accuracies of the dev set.
    :param filename: Name of the file that will save the plot.
    """
    x_axis = [epoch+1 for epoch in range(len(train_accuracy))]
    plt.plot(x_axis, train_accuracy, '-g', linewidth=1, label='Train')
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    plt.plot(x_axis, dev_accuracy, 'b-', linewidth=1, label='Dev')
    plt.legend()
    plt.savefig("plots/"+filename)
    plt.gcf().clear()

class OCRSeqDataset(utils.Dataset):
    """ OCR sequence Dataset """
    def __init__(self, sequences, words, cuda=False):
        """
        :param sequences: Numpy ndarray in which each entry i is a matrix of size Sx128 (S = length of the ith sequence).
        :param words: Numpy ndarray in which each entry i is the target word of the ith sequence, represented as a string.
        """
        letter2ix = dict(zip(string.ascii_lowercase, [i for i in range(26)]))
        self.sequences = [torch.FloatTensor(images) for images in sequences]
        self.targets = [torch.tensor([letter2ix[letter] for letter in word]) for word in words]

    def __getitem__(self, index):
        return (self.sequences[index], self.targets[index])

    def __len__(self):
        return len(self.targets)

