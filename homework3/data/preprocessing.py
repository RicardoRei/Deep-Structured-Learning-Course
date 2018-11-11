# -*- coding: utf-8 -*-
from sklearn.externals import joblib
from tqdm import tqdm
import numpy as np
import argparse
import string


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

def main(sequence=False):
    data = []
    filepointer = open("data/letter.data", 'r')
    for line in tqdm(filepointer):
        line_split = line.split('\t')
        data.append({"id": int(line_split[0]),
                     "letter": line_split[1],
                     "next_id": int(line_split[2]),
                     "word_id": int(line_split[3]),
                     "position": int(line_split[4]),
                     "fold": int(line_split[5]),
                     "features": list(map(int, line_split[6:-1]))})
    if sequence:
        word_data = []
        id2word = {}
        for i in tqdm(range(len(data))):
            if data[i]["word_id"] not in id2word:
                word_info = get_word(i, data)
                word_data.append(word_info)
                id2word[word_info["word_id"]] = word_info["word"]
                
        train_x = [word["features"] for word in word_data if word["fold"] < 8]
        train_y = [word["word"] for word in word_data if word["fold"] < 8]
        dev_x = [word["features"]for word in word_data if word["fold"] == 8]
        dev_y = [word["word"] for word in word_data if word["fold"] == 8]
        test_x = [word["features"] for word in word_data if word["fold"] > 8]
        test_y = [word["word"] for word in word_data if word["fold"] > 8]
        joblib.dump((np.array(train_x, dtype=np.float32), np.array(train_y, dtype=np.long)), 'data/seq_train.pkl')
        joblib.dump((np.array(dev_x, dtype=np.float32), np.array(dev_y, dtype=np.long)), 'data/seq_dev.pkl')
        joblib.dump((np.array(test_x, dtype=np.float32), np.array(test_y, dtype=np.long)), 'data/seq_test.pkl')
    else:
        for sample in data:
            sample["features"] = np.array(np.split(np.array(sample["features"]), 8))
        letter2ix = dict(zip(string.ascii_lowercase, [i for i in range(26)]))
        train_x = [sample["features"] for sample in data if sample["fold"] < 8]
        train_y = [letter2ix[sample["letter"]] for sample in data if sample["fold"] < 8]
        dev_x = [sample["features"] for sample in data if sample["fold"] == 8]
        dev_y = [letter2ix[sample["letter"]] for sample in data if sample["fold"] == 8]
        test_x = [sample["features"] for sample in data if sample["fold"] > 8]
        test_y = [letter2ix[sample["letter"]] for sample in data if sample["fold"] > 8]
        joblib.dump((np.array(train_x, dtype=np.float32), np.array(train_y, dtype=np.long)), 'data/train.pkl')
        joblib.dump((np.array(dev_x, dtype=np.float32), np.array(dev_y, dtype=np.long)), 'data/dev.pkl')
        joblib.dump((np.array(test_x, dtype=np.float32), np.array(test_y, dtype=np.long)), 'data/test.pkl')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=bool, default=False, help="Flag to preprocess the data to feed a sequence model.")
    args = parser.parse_args()
    main(args.sequence)
