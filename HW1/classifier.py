# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

class Classifier(object):
	""" Multi Class Classifier base class """

	def __init__(self, input_size, n_classes):
		"""
		Initializes a matrix in which each column will be the Weights for a specific class.
		:param input_size: Number of features
		:param n_classes: Number of classes to classify the inputs
		"""
		self.parameters = np.zeros((input_size+1, n_classes)) # input_size +1 to include the Bias term

	def train(self, X, Y, devX, devY, epochs=20):
		"""
		This trains the perceptron over a certain number of epoch and records the accuracy in Train and Dev sets along each epoch.
		:param X: numpy array with size DxN where D is the number of training examples and N is the number of features.
		:param Y: numpy array with size D containing the correct labels for the training set
		:param devX (optional): same as X but for the dev set.
		:param devY (optional): same as Y but for the dev set.
		:param epochs (optional): number of epochs to run

		Note: This function will print a loading bar ate the terminal for each epoch.
		"""
		train_accuracy = [self.evaluate(X, Y)]
		dev_accuracy = [self.evaluate(devX, devY)]
		for epoch in range(epochs):
			for i in tqdm(range(X.shape[0])):
				self.update_weights(X[i, :], Y[i])
			train_accuracy.append(self.evaluate(X, Y))
			dev_accuracy.append(self.evaluate(devX, devY))
		return train_accuracy, dev_accuracy
		

	def evaluate(self, X, Y):
		"""
		Evaluates the error in a given set of examples.
		:param X: numpy array with size DxN where D is the number of examples to evaluate and N is the number of features.
		:param Y: numpy array with size D containing the correct labels for the training set
		"""
		correct_predictions = 0
		for i in range(X.shape[0]):
			y_pred = self.predict(X[i, :])
			if Y[i] == y_pred:
				correct_predictions += 1
		return correct_predictions/X.shape[0]


	def plot_train(self, train_accuracy, dev_accuracy):
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


	def update_weights(self, x, y):
		"""
		Function that will take an input example and the true prediction and will update the model parameters.
		:param x: Array of size N where N its the number of features that the model takes as input.
		:param y: The int corresponding to the correct label.

		TO DO - child classes must implement this function
		"""
		pass

	def predict(self, x):
		"""
		This function will add a Bias value to the received input, multiply the Weights corresponding to the different classes
		with the input vector and choose the class that maximizes that multiplication.
		:param x: numpy array with size 1xN where N = number of features.

		TO DO - child classes must implement this function
		"""
		pass
