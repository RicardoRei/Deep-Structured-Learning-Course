# -*- coding: utf-8 -*-
from sklearn.externals import joblib
from Classifier import Classifier
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class MultinomialLR(Classifier):
	""" Multinomial Logistic Regression """

	def __init__(self, input_size, n_classes, lr=0.001, l2=None):
		"""
		Initializes a matrix in which each column will be the Weights for a specific class.
		:param input_size: Number of features
		:param n_classes: Number of classes to classify the inputs
		"""
		Classifier.__init__(self, input_size, n_classes)
		self.lr = lr
		self.l2 = l2

	def predict(self, input):
		"""
		This function will add a Bias value to the received input, multiply the Weights corresponding to the different classes
		with the input vector, run a softmax function and choose the class that achieves an higher probability.
		:param x: numpy array with size 1xN where N = number of features.
		"""
		return np.argmax(self.softmax(np.dot(np.append(input, [1]), self.parameters)))

	def softmax(self, x):
		""" Compute softmax values for each sets of scores in x."""
		return np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x)), axis=0)

	def update_weights(self, x, y):
		"""
		Function that will take an input example and the true prediction and will update the model parameters.
		:param x: Array of size N where N its the number of features that the model takes as input.
		:param y: The int corresponding to the correct label.
		"""
		linear = np.dot(np.append(x, [1]), self.parameters)
		predictions = self.softmax(linear)

		if self.l2 is not None:
			self.parameters = self.parameters - self.lr*(np.outer(predictions, np.append(x, [1])).T - self.l2*self.parameters)
		else: 
			self.parameters = self.parameters - self.lr*(np.outer(predictions, np.append(x, [1])).T)
		self.parameters[:, y] = self.parameters[:, y] + self.lr*np.append(x, [1])

def main():
	train_x, train_y = joblib.load("data/train.pkl")
	dev_x, dev_y = joblib.load("data/dev.pkl")
	test_x, test_y = joblib.load("data/test.pkl")
	logistic_reg = MultinomialLR(train_x.shape[1], np.unique(train_y).shape[0], l2=0.1)
	train_accuracy, dev_accuracy = logistic_reg.train(train_x, train_y, dev_x, dev_y)
	print ("Train Accuracy: {}".format(logistic_reg.evaluate(train_x, train_y)))
	print ("Dev Accuracy: {}".format(logistic_reg.evaluate(dev_x, dev_y)))
	print ("Test Accuracy: {}".format(logistic_reg.evaluate(test_x, test_y)))
	logistic_reg.plot_train(train_accuracy, dev_accuracy)
	
if __name__ == '__main__':
	main()


