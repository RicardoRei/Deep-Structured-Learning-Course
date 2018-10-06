# -*- coding: utf-8 -*-
from sklearn.externals import joblib
from Classifier import Classifier
import matplotlib.pyplot as plt
import numpy as np

class MultinomialLR(Classifier):
	""" Multinomial Logistic Regression """

	def __init__(self, input_size, n_classes, lr=0.001):
		"""
		Initializes a matrix in which each column will be the Weights for a specific class.
		:param input_size: Number of features
		:param n_classes: Number of classes to classify the inputs
		"""
		Classifier.__init__(self, input_size, n_classes)
		self.lr = lr

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
		for c in range(self.parameters.shape[1]):
			self.parameters[:, c] = self.parameters[:, c] + self.lr*((c == y)*np.append(x, [1]) - predictions[c]*np.append(x, [1]))

def main():
	train_x, train_y = joblib.load("data/kernel_train.pkl")
	dev_x, dev_y = joblib.load("data/kernel_dev.pkl")
	test_x, test_y = joblib.load("data/kernel_dev.pkl")
	logistic_reg = MultinomialLR(train_x.shape[1], np.unique(train_y).shape[0])
	train_accuracy, dev_accuracy = logistic_reg.train(train_x, train_y, devX=dev_x, devY=dev_y)
	print (logistic_reg.evaluate(train_x, train_y))
	print (logistic_reg.evaluate(dev_x, dev_y))
	print (logistic_reg.evaluate(test_x, test_y))
	logistic_reg.plot_train(train_accuracy, dev_accuracy)
	
if __name__ == '__main__':
	main()


