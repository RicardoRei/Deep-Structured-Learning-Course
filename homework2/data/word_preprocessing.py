# -*- coding: utf-8 -*-
from sklearn.externals import joblib
from tqdm import tqdm
import numpy as np

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

def get_word(initial_idx, data):
	i = initial_idx
	word_features = []
	word = ""
	while (int(data[i]["next_id"]) != -1):
		word += data[i]["letter"]
		word_features.append(data[i]["features"])
		i += 1
	word += data[i]["letter"]
	word_features.append(data[i]["features"])

	return {"word": word, 
			"word_id": data[i]["word_id"], 
			"fold": data[i]["fold"], 
			"features": np.array(word_features)}
	
def main():
	data = []

	filepointer = open("letter.data", 'r')
	for line in tqdm(filepointer):
		line_split = line.split('\t')
		data.append({"id": line_split[0],
					 "letter": line_split[1],
					 "next_id": line_split[2],
					 "word_id": line_split[3],
					 "position": line_split[4],
					 "fold": int(line_split[5]),
					 "features": list(map(int, line_split[6:-1]))
			})
	filepointer.close()
	word_data = []
	id2word = {}
	for i in tqdm(range(len(data))):
		if data[i]["word_id"] not in id2word:
			word_info = get_word(i, data)
			word_data.append(word_info)
			id2word[word_info["word_id"]] = word_info["word"]
	
	train_x = [np.array(word["features"]) for word in word_data if word["fold"] < 8]
	train_y = [word["word"] for word in word_data if word["fold"] < 8]

	dev_x = [np.array(word["features"]) for word in word_data if word["fold"] == 8]
	dev_y = [word["word"] for word in word_data if word["fold"] == 8]

	test_x = [np.array(word["features"]) for word in word_data if word["fold"] > 8]
	test_y = [word["word"] for word in word_data if word["fold"] > 8]

	print ("train size: {}\ndev size: {}\ntest size:{}".format(len(train_x), len(dev_x), len(test_x)))
	joblib.dump((np.array(train_x), np.array(train_y)), 'structured_train.pkl')
	joblib.dump((np.array(dev_x), np.array(dev_y)), 'structured_dev.pkl')
	joblib.dump((np.array(test_x), np.array(test_y)), 'structured_test.pkl')


if __name__ == '__main__':
	main()

