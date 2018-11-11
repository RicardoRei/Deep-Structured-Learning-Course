# -*- coding: utf-8 -*-
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *
import numpy as np
import string


class StructuredPerceptron(object):
		""" Structured Perceptron class: 
		This class was implemented for the OCR dataset: http://ai.stanford.edu/~btaskar/ocr/ 
		"""
	def __init__(self, input_size, n_classes, pairwise=True):
		"""
		:param input_size: Number of features
		:param n_classes: Number of classes to classify the inputs
		:param pairwise: Flag that applies pairwise multiplication to the features.
		"""
		self.pairwise = pairwise
		self.n_classes = n_classes
		self.input_size = input_size*input_size if pairwise else input_size
		# Unigram Parameters matrix: each line is related with a class/character.
		self.Wunig= np.zeros((n_classes, self.input_size))
		# Bigram Parameters matrix: similar to the transition matrix in an HMM.
		self.Wbig = np.zeros((n_classes, n_classes))
		# Start score array: similar to the initial distribution in an HMM.
		self.Wstart = np.zeros(n_classes)
		# Start score array: similar to the final distribution in an HMM.
		self.Wstop = np.zeros(n_classes)

	def forward(self, input_seq):
		"""
		Forward propagation over the sequence passed as input.
		:param input_seq: Matrix in which each line is the pixel values of an image.
		"""
		backtrack_matrix = np.zeros((self.n_classes, input_seq.shape[0]-1))
		# The initial score is given by Unigram features and the start parameters.
		
		unigram_score = np.dot(self.Wunig, self.pixel_pairwise(input_seq[0, :])) \
		if self.pairwise else np.dot(self.Wunig, input_seq[0, :])
		bigram_score = self.Wstart
		y_hat = unigram_score + bigram_score

		# Intermediate Positions
		for i in range(1, input_seq.shape[0]):
			unigram_score = np.dot(self.Wunig, self.pixel_pairwise(input_seq[i, :]))
			y_hat = self.Wbig + np.vstack((y_hat for j in range(self.n_classes))).T
			backtrack_matrix[:, i-1] = np.argmax(y_hat, axis=0)
			y_hat = np.max(y_hat, axis=0) + unigram_score

		# Final Position
		last_score =  self.Wstop + y_hat
		last_state = int(np.argmax(last_score))

		# Recover sequence
		states = [last_state, ]
		for i in range(backtrack_matrix.shape[1]-1, -1, -1):
			states.insert(0, int(backtrack_matrix[:, i][int(states[0])]))
		return states

	def update_weights(self, x, y):
		"""
		Function that will take a sequence of digits, runs a forward propagation and updates the weights according 
		to the obtained sequence.
		:param x: Matrix in which each line is the pixel values of an image.
		:param y: List containing the index of the target letters for the inputed sequence.
		"""
		pred_states = self.forward(x)
		# Update start scores
		if pred_states[0] != y[0]:
			self.Wstart[y[0]] += 1
			self.Wstart[pred_states[0]] -= 1
			self.Wunig[y[0]] += self.pixel_pairwise(x[0]) \
			if self.pairwise else x[0]
			self.Wunig[pred_states[0]] -= self.pixel_pairwise(x[0]) \
			if self.pairwise else x[0]

		# Update Intermediate Scores
		for i in range(1, len(pred_states)):
			if pred_states[i] != y[i]:
				self.Wunig[y[i]] += self.pixel_pairwise(x[i]) 
				self.Wunig[pred_states[i]] -= self.pixel_pairwise(x[i, :]) 

				self.Wbig[pred_states[i-1], pred_states[i]] -= 1
				self.Wbig[y[i-1], y[i]] += 1

		# Update Final Scores
		if pred_states[-1] != y[-1]:
			self.Wstop[y[-1]] += 1
			self.Wstop[pred_states[-1]] -= 1

	def train(self, X, Y, devX, devY, epochs=20):
		"""
		This trains the perceptron over a certain number of epoch and records the accuracy in Train and Dev sets along each epoch.
		:param X: numpy array with size DxN where D is the number of training examples and N is the number of features.
		:param Y: numpy array with size D containing the correct labels for the training set
		:param devX: same as X but for the dev set.
		:param devY: same as Y but for the dev set.
		:param epochs (optional): number of epochs to run

		Note: This function will print a loading bar ate the terminal for each epoch.
		"""
		train_accuracy = [self.evaluate(X, Y)]
		dev_accuracy = [self.evaluate(devX, devY)]
		for epoch in range(epochs):
			for i in tqdm(range(X.shape[0])):
				self.update_weights(X[i], Y[i])
			train_accuracy.append(self.evaluate(X, Y))
			dev_accuracy.append(self.evaluate(devX, devY))
			print ("Train Accuracy: {}\nDev Accuracy: {}".format(train_accuracy[-1] ,dev_accuracy[-1]))
		return train_accuracy, dev_accuracy

	def evaluate(self, X, Y):
		"""
		Evaluates the error in a given set of examples.
		:param X: numpy array with size DxN where D is the number of examples to evaluate and N is the number of features.
		:param Y: numpy array with size D containing the correct labels for the training set
		"""
		correct_predictions = 0
		total = 0
		for i in range(X.shape[0]):
			pred_states = self.forward(X[i])
			for j in range(len(pred_states)):
				total += 1
				if pred_states[j] == Y[i][j]:
					correct_predictions += 1
		return correct_predictions/total

	def pixel_pairwise(self, pixel_array):
		"""
		Multiplies each entry Xi with all the others to create a new array of pairwise multiplications.
		:param pixel_array: Numpy array with the original pixels.
		"""
		return np.outer(pixel_array, pixel_array).flatten()


def main():
	train_x, train_y = joblib.load("data/structured_train.pkl")
	dev_x, dev_y = joblib.load("data/structured_dev.pkl")
	letter2ix = dict(zip(string.ascii_lowercase, [i for i in range(26)]))
	test_x, test_y = joblib.load("data/structured_test.pkl")
	perceptron = StructuredPerceptron(128, 26, pairwise=True)
	train_accuracy, dev_accuracy = perceptron.train(train_x, [[letter2ix[ix] for ix in train_y[i]] for i in range(train_y.shape[0])], dev_x, [[letter2ix[ix] for ix in dev_y[i]] for i in range(dev_y.shape[0])], epochs=20)
	print ("Test Accuracy: {}".format(perceptron.evaluate(test_x, [[letter2ix[ix] for ix in test_y[i]] for i in range(test_y.shape[0])])))
	plot_train(train_accuracy, dev_accuracy)

if __name__ == '__main__':
	main()