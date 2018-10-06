# Homework 1:
File named homework1.pdf contains the homework description.

### Preprocessing:
Folder data contains the data (file letter.data) and a file named preprocessing.py that will generate the numpy matrixes containing the classifiers inputs and the targets.
<br />
To run the preprossing script you just need to type: python3 preprocessing.py
Optionally if you want to test the Linear Kernel feature Transfomation you can type: python3 preprocessing.py --kernel=True
After running the preprocessing python script you can load the data using the sklearn joblib function. (E.g: train_x, train_y = joblib.load("data/kernel_train.pkl"))

### Classifiers:
To run the classifiers you only need to call the respective file. (E.g: running the Perceptron: python3 Perceptron.py)

## Requirements:
python 3.6.6
numpy 1.15.2
matplotlib 3.0.0
sklearn 0.20.0
tqdm 4.26.0

