# -*- coding: utf-8 -*-
from sklearn.externals import joblib
from Classifier import Classifier
import matplotlib.pyplot as plt
import numpy as np

class MultiClassPerceptron(Classifier):
	""" Multi Class Perceptron """

	def __init__(self, input_size, n_classes):
		"""
		Initializes a matrix in which each column will be the Weights for a specific class.
		:param input_size: Number of features
		:param n_classes: Number of classes to classify the inputs
		"""
		Classifier.__init__(self, input_size, n_classes)

	def predict(self, x):
		"""
		This function will add a Bias value to the received input, multiply the Weights corresponding to the different classes
		with the input vector and choose the class that maximizes that multiplication.
		:param x: numpy array with size 1xN where N = number of features.
		"""
		return np.argmax(np.dot(np.append(x, [1]), self.parameters))

	def update_weights(self, x, y):
		"""
		Function that will take an input example and the true prediction and will update the model parameters.
		:param x: Array of size N where N its the number of features that the model takes as input.
		:param y: The int corresponding to the correct label.
		"""
		y_pred = self.predict(x)
		if y != y_pred:
			self.parameters[:, y] += np.append(x, [1])
			self.parameters[:, y_pred] -=  np.append(x, [1])

def main():
	train_x, train_y = joblib.load("data/kernel_train.pkl")
	dev_x, dev_y = joblib.load("data/kernel_dev.pkl")
	perceptron = MultiClassPerceptron(train_x.shape[1], np.unique(train_y).shape[0])
	train_accuracy, dev_accuracy = perceptron.train(train_x, train_y, devX=dev_x, devY=dev_y)
	perceptron.plot_train(train_accuracy, dev_accuracy)

if __name__ == '__main__':
	main()


