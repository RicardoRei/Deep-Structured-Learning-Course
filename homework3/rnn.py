# -*- coding: utf-8 -*-
from sklearn.externals import joblib
from utils import *
import argparse

# ----------------------------------------------------------------------------------------------------------------
#                                             MODELS                                                                |
# ----------------------------------------------------------------------------------------------------------------
class BiLSTM(nn.Module):
    """ OCR BiLSTM for exercise 2.1 of homework 3. This module takes as input a sequence of flatten image (128 dims),
        runs a feed-forward layer and then it passes the computed representations to a BiLSTM.  """
    def __init__(self, input_size, n_classes, hidden_size, num_layers=2, dropout=0., activation="Sigmoid"):
        """
        :param input_size: Number of input feature (128).
        :param n_classes: Number of classes (26).
        :param hidden_size: Size of the LSTM hidden layers.
        :param num_layers: Number of hidden layers.
        :param activation: Activation function top be used in the forward layer.
                           (see torch.nn documentation to see available activations).
        """
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.n_classes = n_classes
        self.num_layers = num_layers
        self.ff_affine = nn.Linear(self.input_size, self.hidden_size)
        activation = getattr(torch.nn, activation)
        self.ff_activation = activation()
        if num_layers > 2:
            self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, bias=True, batch_first=True, dropout=dropout, bidirectional=True)
        else:
            self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, bias=True, batch_first=True, bidirectional=True)
        self.affine = nn.Linear(hidden_size*2, n_classes)

    def forward(self, sequence):
        """
        :param sequence: Tensor of size LxN where L is the length of the sequece and N is the number of features.
        """
        # Run a feed forward layer for each sequence of images inside the batch.
        unigram_features = self.ff_activation(self.ff_affine(sequence))
        # Since not every sequence has an equal size we need to pad some sequences before running the lstm.
        outputs, (h_n, _) = self.lstm(unigram_features)
        # Apply one last affine transformation
        return self.affine(outputs)[0]


class ConvLSTM(nn.Module):
    """ OCR BiLSTM for exercise 2.3 of homework 3. This module takes as input a sequence of flatten image (128 dims),
        transforms it into a matrix 8x16, runs a convolution layer and then it passes the computed representation to
        a BiLSTM.  
    """
    def __init__(self, input_size, n_classes, hidden_size, num_layers=2, dropout=0., activation="Sigmoid"):
        """
        :param input_size: Number of input feature (128).
        :param n_classes: Number of classes (26).
        :param hidden_size: Size of the LSTM hidden layers.
        :param num_layers: Number of hidden layers.
        :param activation: Activation function (from torch.nn module) top be used in the forward layer.
                           (see torch.nn documentation to see available activations).
        """
        super(ConvLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.n_classes = n_classes
        self.num_layers = num_layers
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, padding=2, stride=1, bias=True)
        if num_layers > 1:
            self.lstm = nn.LSTM(self.input_size*20, self.hidden_size, self.num_layers, bias=True, batch_first=True, dropout=dropout, bidirectional=True)
        else:
            self.lstm = nn.LSTM(self.input_size*20, self.hidden_size, self.num_layers, bias=True, batch_first=True, bidirectional=True)
        self.affine = nn.Linear(hidden_size*2, n_classes)

    def forward(self, sequence):
        """
        :param sequence: Tensor of size LxN where L is the length of the sequece and N is the number of features.
        """
        # Transform the inputs back to the original image 8x16
        sequence = sequence.view(sequence.shape[1], sequence.shape[0], 8, 16)
        #sequence = torch.unsqueeze(sequence, 1)
        unigram_features = self.conv_layer(sequence)
        unigram_features = unigram_features.view(-1, 20*8*16).unsqueeze(0)
        # Since not every sequence has an equal size we need to pad some sequences before running the lstm.
        outputs, (h_n, _) = self.lstm(unigram_features)
        # Apply one last affine transformation
        return self.affine(outputs)[0]
        
# ----------------------------------------------------------------------------------------------------------------
#                                              Main                                                                 |
# ----------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Epochs to run.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate to be used.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout to be applied between hidden layers (ignored if num_layers < 2).")
    parser.add_argument("--optim", type=str, default="Adam", help="Optimizer function name.")
    parser.add_argument("--cuda", type=bool, default=True, help="Flag to train the model in CUDA.")
    parser.add_argument("--hidden_size", type=int, default=256, help="Size of the LSTM hiddens.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the LSTM.")
    parser.add_argument("--model", type=str, default="BiLSTM", help="Model class to be used.")
    args = parser.parse_args()

    # Load data.
    train_x, train_y = joblib.load("data/seq_train.pkl")
    dev_x, dev_y = joblib.load("data/seq_dev.pkl")
    test_x, test_y = joblib.load("data/seq_test.pkl")
    train_data = OCRSeqDataset(train_x, train_y)
    dev_data = OCRSeqDataset(dev_x, dev_y)
    test_data = OCRSeqDataset(test_x, test_y)
    train_loader = utils.DataLoader(train_data, shuffle=True)
    dev_loader = utils.DataLoader(dev_data, shuffle=True)
    test_loader = utils.DataLoader(test_data, shuffle=True)

    # Create Model.
    if args.model == "ConvLSTM":
        lstm = ConvLSTM(train_x[0].shape[1], 26, args.hidden_size, args.num_layers, args.dropout)
    else:
        lstm = BiLSTM(train_x[0].shape[1], 26, args.hidden_size, args.num_layers, args.dropout)

    print ("Model: {}".format(lstm.__class__.__name__))
    lstm, train_accuracy, dev_accuracy, losses = train(
        lstm, train_loader, dev_loader, optim=args.optim, epochs=args.epochs, lr=args.lr, cuda=args.cuda
    )

    plot_loss(losses, "{}-losses.png".format(args.model))
    plot_train(train_accuracy, dev_accuracy, "{}-train-accuracy.png".format(args.model))
    print ("Test Accuracy: {0:.6f}".format(evaluate(lstm, test_loader)))

if __name__ == '__main__':
    main()
