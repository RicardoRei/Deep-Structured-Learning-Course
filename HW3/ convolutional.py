# -*- coding: utf-8 -*-
from sklearn.externals import joblib
import torch.nn.functional as F
from utils import *
import numpy as np
import argparse
import torch

class ConvolutionalNN(nn.Module):
	def __init__(self, input_size, n_classes):
		super(ConvolutionalNN, self).__init__()
		pass

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dropout", type=float, default=0.0, help="Dropout value to apply during train.")
	parser.add_argument("--epochs", type=int, default=20, help="Epochs to run.")
	parser.add_argument("--lr", type=float, default=0.001, help="Learning rate to be used.")
	parser.add_argument("--optim", type=str, default="Adam", help="Optimizer function name.")
	args = parser.parse_args()
	
	
if __name__ == '__main__':
	main()