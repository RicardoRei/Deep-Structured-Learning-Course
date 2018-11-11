# -*- coding: utf-8 -*-
from sklearn.externals import joblib
import torch.nn.functional as F
from utils import *
import numpy as np
import argparse
import torch


class LogisticRegression(nn.Module):
	""" LogisticRegression: Pytorch implementation of a Logistic Regression module. """
	
	def __init__(self, input_size, n_classes):
		"""
		:param input_size: Input size expected.
		:param n_classes: Number of classes.
		"""
		super(LogisticRegression, self).__init__()
		self.linear = nn.Linear(input_size, n_classes)
		self.linear.weight.data = torch.zeros(self.linear.weight.shape)

	def forward(self, x):
		"""
		Returns the softmax distribution of x over the set of classes.
		:param x: Input that we want to classify.
		"""
		return F.softmax(self.linear(torch.FloatTensor(x).view(1,-1)), dim=1)

	def predict(self, x):
		"""
		Predicts the most likely class for x.
		:param x: Input that we want to classify.
		"""
		with torch.no_grad():
			return np.argmax(self.forward(x).numpy())

	def train_mode(self):
		""" Not used. """
		pass

	def test_mode(self):
		""" Not used. """
		pass

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", type=int, default=10, help="Epochs to run.")
	parser.add_argument("--lr", type=float, default=0.001, help="Learning rate to be used.")
	parser.add_argument("--optim", type=str, default="Adam", help="Optimizer function name.")
	parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay constant to be used (Regularization constant).")
	args = parser.parse_args()

	# Load data
	train_x, train_y = joblib.load("data/train.pkl")
	dev_x, dev_y = joblib.load("data/dev.pkl")
	test_x, test_y = joblib.load("data/test.pkl")
	
	# initialize model
	logistic_reg = LogisticRegression(train_x.shape[1], np.unique(train_y).shape[0])

	# start training
	logistic_reg, train_accuracy, dev_accuracy = train(
		logistic_reg, train_x, train_y, dev_x, dev_y, 
		weight_decay=args.weight_decay, optim=args.optim, lr=args.lr, epochs=args.epochs
	)

	# plot accuracies during training.
	plot_train(train_accuracy, dev_accuracy)
	
if __name__ == '__main__':
	main()




	



