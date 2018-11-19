# -*- coding: utf-8 -*-
from sklearn.externals import joblib
from tqdm import tqdm
import numpy as np
import argparse
import pickle
import os

"""
The file containing the binary pixel images is organized as follows:

    id: each letter is assigned a unique integer id
    letter: a-z
    next_id: id for next letter in the word, -1 if last letter
    word_id: each word is assigned a unique integer id (not used)
    position: position of letter in the word (not used)
    fold: 0-9 -- cross-validation fold
    p_i_j: 0/1 -- value of pixel in row i, column j

from the previous attributes we will only need the id, letter, fold and the pixel values.
"""


def pixel_pairwise_combination(pixel_array):
    """
    Multiplies each entry Xi with all the others to create a new array of pairwise multiplications.
    :param pixel_array: Numpy array with the original pixels.
    """
    return np.outer(pixel_array, pixel_array).flatten()


def main(pairwise=False):
    letter2idx = {}
    train_y = []
    train_x = []

    dev_y = []
    dev_x = []

    test_y = []
    test_x = []

    filepointer = open("data/letter.data", 'r')
    for line in tqdm(filepointer):
        line_split = line.split('\t')
        if line_split[1] not in letter2idx:
            letter2idx[line_split[1]] = len(letter2idx)
        if int(line_split[5]) < 8:
            train_y.append(letter2idx[line_split[1]])
            train_x.append(np.array(list(map(int, line_split[6:-1]))))
        elif int(line_split[5]) == 8:
            dev_y.append(letter2idx[line_split[1]])
            dev_x.append(np.array(list(map(int, line_split[6:-1]))))
        else:
            test_y.append(letter2idx[line_split[1]])
            test_x.append(np.array(list(map(int, line_split[6:-1]))))
    filepointer.close()
    if pairwise:
        for i in tqdm(range(len(train_x))):
            train_x[i] = pixel_pairwise_combination(train_x[i])
        for i in tqdm(range(len(dev_x))):
            dev_x[i] = pixel_pairwise_combination(dev_x[i])
        for i in tqdm(range(len(test_x))):
            test_x[i] = pixel_pairwise_combination(test_x[i])
    joblib.dump((np.array(train_x), np.array(train_y)), 'data/train.pkl')
    joblib.dump((np.array(dev_x), np.array(dev_y)), 'data/dev.pkl')
    joblib.dump((np.array(test_x), np.array(test_y)), 'data/test.pkl')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairwise", type=bool, default=False, help="Flag to apply the pairwise multiplication transformation to the original pixels.")
    args = parser.parse_args()
    main(args.pairwise)

