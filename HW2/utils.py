# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch

def train(model, X, Y, devX, devY, optim="SGD", lr=0.01, weight_decay=0.0, epochs=5):
	"""
	This trains the perceptron over a certain number of epoch and records the accuracy in Train and Dev sets along each epoch.
	:param model: Pytorch model to be trained.
	:param X: numpy array with size DxN where D is the number of training examples and N is the number of features.
	:param Y: numpy array with size D containing the correct labels for the training set
	:param devX  (optional): same as X but for the dev set.
	:param devY  (optional): same as Y but for the dev set.
	:param optim (optional): Name of the optimizer used during training.
	:param lr 	 (optional): learning rate to be used.
	:param weight_decay (optional): Regularization constant.
	:param epochs (optional): number of epochs to run

	Note: This function will print a loading bar ate the terminal for each epoch.
	"""
	optim = getattr(torch.optim, optim)
	optimizer = optim(model.parameters(), lr=lr, weight_decay=weight_decay)
	loss_func = nn.CrossEntropyLoss()
	model.test_mode()
	train_accuracy = [evaluate(model, X, Y)]
	dev_accuracy = [evaluate(model, devX, devY)]
	print ("Validation set Accuracy: {0:.4f}".format(dev_accuracy[0]))
	print ("Starting training...")
	for epoch in range(epochs):
		# Shuffle dataset
		X, Y = shuffle_data(X, Y)
		model.train_mode() # Activate dropout
		total_loss = 0
		for i in tqdm(range(X.shape[0])):
			# clear gradients
			model.zero_grad()
			# compute probabilities
			probs = model(X[i])
			target = torch.tensor([Y[i]], dtype=torch.long)
			loss = loss_func(probs, target)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		model.test_mode() # Deactivate dropout
		train_accuracy.append(evaluate(model, X, Y))
		dev_accuracy.append(evaluate(model, devX, devY))
		print ("Loss: {0:.4f}\nValidation set Accuracy: {1:.3f}\nTrain set Accuracy: {2:.3f}".format(total_loss / X.shape[0], dev_accuracy[-1], train_accuracy[-1]))
	return model, train_accuracy, dev_accuracy

def shuffle_data(X, Y):
	"""
	Function that takes the all dataset and shuffles examples.
	:param X: Inputs to be shuffled.
	:param Y: Expected targets.
	"""
	perm = np.random.permutation(X.shape[0])
	return X[perm], Y[perm]

def create_batches(X, Y, batch_size=1024):
	"""
	Splits the dataset into batches of a given size.
	:param X: Inputs to be splitted.
	:param Y: Expected targets.
	"""
	divisor = X.shape[0]/batch_size
	return np.array_split(X, divisor), np.array_split(Y, divisor)

def evaluate(model, X, Y):
	"""
	Evaluates the error in a given set of examples.
	:param model: Pytorch model to evaluate.
	:param X: numpy array with size DxN where D is the number of examples to evaluate and N is the number of features.
	:param Y: numpy array with size D containing the correct labels for the training set
	"""
	x_batches, y_batches = create_batches(X, Y)
	correct_predictions = 0
	for i in range(X.shape[0]):
		y_pred = model.predict(X[i])
		if Y[i] == y_pred:
			correct_predictions += 1
	return correct_predictions/X.shape[0]

def plot_train(train_accuracy, dev_accuracy):
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
	plt.show()


