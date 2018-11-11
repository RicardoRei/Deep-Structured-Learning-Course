# -*- coding: utf-8 -*-
from sklearn.externals import joblib
import torch.nn.functional as F
from utils import *
import numpy as np
import argparse
import torch

class FeedForwardNN(nn.Module):
	""" FeedForwardNN: Pytorch implementation of a single layer feed forward neural network. """
	
	def __init__(self, input_size, n_classes, hidden_size, dropout=0, activation_func="Sigmoid"):
		"""
		:param input_size: Input size expected..
		:param n_classes: Number of classes.
		:param hidden_size: Size of the hidden layers.
		:param dropout: dropout to be used after the hidden layer activation.
		:param activation_func: Name of the activation function to be used 
								(see torch.nn documentation to see available activations).
		"""
		super(FeedForwardNN, self).__init__()
		self.dropout_value = dropout
		self.dropout = nn.Dropout(p=dropout)
		self.l0_linear = nn.Linear(input_size, hidden_size)
		self.l1_linear = nn.Linear(hidden_size, n_classes)
		activation = getattr(nn, activation_func)
		self.activation = activation()

	def forward(self, x):
		"""
		Returns the softmax distribution of x over the set of classes.
		:param x: Input that we want to classify.
		"""
		layer0_out = self.activation(self.l0_linear(torch.FloatTensor(x).view(1,-1)))
		return F.softmax(self.dropout(self.l1_linear(layer0_out)), dim=1)

	def predict(self, x):
		"""
		Predicts the most likely class for x.
		:param x: Input that we want to classify.
		"""
		with torch.no_grad():
			return np.argmax(self.forward(x).numpy())

	def train_mode(self):
		"""
		Sets dropout to the original value.
		"""
		self.dropout.p = self.dropout_value

	def test_mode(self):
		"""
		Sets dropout to zero (used during inference).
		"""
		self.dropout.p = 0.0


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dropout", type=float, default=0.0, help="Dropout value to apply during train.")
	parser.add_argument("--epochs", type=int, default=20, help="Epochs to run.")
	parser.add_argument("--lr", type=float, default=0.001, help="Learning rate to be used.")
	parser.add_argument("--activation", type=str, default="Sigmoid", help="Activation function name.")
	parser.add_argument("--optim", type=str, default="Adam", help="Optimizer function name.")
	parser.add_argument("--hidden_size", type=int, default=128, help="Number of hidden layer to consider.")
	args = parser.parse_args()

	# Load data.
	train_x, train_y = joblib.load("data/train.pkl")
	dev_x, dev_y = joblib.load("data/dev.pkl")
	test_x, test_y = joblib.load("data/test.pkl")

	# Initialize model.
	ff_nn = FeedForwardNN(
		train_x.shape[1], np.unique(train_y).shape[0], args.hidden_size, 
		dropout=args.dropout, activation_func=args.activation
	)
	
	# Start training.
	ff_nn, train_accuracy, dev_accuracy = train(
		ff_nn, train_x, train_y, dev_x, dev_y, 
		optim=args.optim, lr=args.lr, epochs=args.epochs
	)

	print (evaluate(ff_nn, test_x, test_y))
	# plot accuracies during training.
	plot_train(train_accuracy, dev_accuracy)
	
if __name__ == '__main__':
	main()




	



