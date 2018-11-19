# -*- coding: utf-8 -*-
from sklearn.externals import joblib
import torch.utils.data as utils
import torch.nn.functional as F
from utils import *
import argparse

# ----------------------------------------------------------------------------------------------------------------
#                                             MODELS                                                                |
# ----------------------------------------------------------------------------------------------------------------
class ConvNN1(nn.Module):
    """ Simple convolution neural network as described in the homework3 exercise."""
    def __init__(self, input_shape, n_classes):
        """
        :param input_shape: Tuple containing the input matrix size.
        :param n_classes: Number of classes we want to classify.
        :param dropout: Dropout to be applied before the softmax.
        """
        super(ConvNN1, self).__init__()
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.activation = nn.ReLU()
        self.first_conv = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, padding=2, stride=1)
        self.first_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.second_conv = nn.Conv2d(in_channels=20, out_channels=30, kernel_size=7, stride=1, padding=[5,7]) 
        self.second_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.affine = nn.Linear(4*8*30, n_classes)

    def forward(self, images):
        """
        :param images: Batch containing several images to be processed.
        """
        images = torch.unsqueeze(images, 1)
        first_conv = self.first_conv(images)
        first_pooling = self.first_pooling(self.activation(first_conv))
        second_conv = self.second_conv(first_pooling)
        second_pooling = self.second_pooling(self.activation(second_conv))
        print (first_conv.shape)
        print (first_pooling.shape)
        print (second_conv.shape)
        print (second_pooling.shape)
        exit()
        second_pooling = second_pooling.view(-1, 4*8*30)
        return self.affine(second_pooling)

    def print_filters(self):
        layer1_weights = self.first_conv.weight.cpu().data.numpy()
        layer2_weights = self.second_conv.weight.cpu().data.numpy()
        random_channels = np.random.randint(0, layer1_weights.shape[0], 3)
        while not np.unique(random_channels).shape == random_channels.shape:
            random_channels = np.random.randint(0, layer1_weights.shape[0], 3)
        for i in random_channels:
            random_filter = layer1_weights[i, 0, :]
            plt.imshow(random_filter, cmap='magma', interpolation='nearest')
            plt.savefig("plots/cnn-layer-1-filter{}-magma.png".format(i), bbox_inches='tight')
            plt.imshow(random_filter, cmap='gray', interpolation='nearest')
            plt.savefig("plots/cnn-layer-1-filter{}-grey.png".format(i), bbox_inches='tight')
        random_channels = np.random.randint(0, layer2_weights.shape[0], 3)
        while not np.unique(random_channels).shape == random_channels.shape:
            random_channels = np.random.randint(0, layer2_weights.shape[0], 3)

        random_depth = np.random.randint(0, layer2_weights.shape[1], 3)
        for i in range(3):
            random_filter = layer2_weights[random_channels[i], random_depth[i], :]
            plt.imshow(random_filter, cmap='magma', interpolation='nearest')
            plt.savefig("plots/cnn-layer-2-filter{}-magma.png".format(i), bbox_inches='tight')
            plt.imshow(random_filter, cmap='gray', interpolation='nearest')
            plt.savefig("plots/cnn-layer-2-filter{}-grey.png".format(i), bbox_inches='tight')

class ConvNN2(nn.Module):
    """ This model is similar to the last one but with a multi-layer perceptron on top. """
    def __init__(self, input_shape, n_classes, mlp_hidden_units=256):
        """
        :param input_shape: Tuple containing the input matrix size.
        :param n_classes: Number of classes we want to classify.
        :param mlp_hidden_units: Number of hidden units in the MLP.
        :param dropout: Dropout to be applied before the softmax.
        """
        super(ConvNN2, self).__init__()
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.activation = nn.ReLU()
        self.first_conv = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, padding=2, bias=True)
        self.first_pooling = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.second_conv = nn.Conv2d(in_channels=20, out_channels=30, kernel_size=(7,7), padding=3, bias=True) 
        self.second_pooling = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.mlp_l0_linear = nn.Linear(2*4*30, mlp_hidden_units)
        self.mlp_activation = nn.Sigmoid()
        self.mlp_l1_linear = nn.Linear(mlp_hidden_units, n_classes)
        
    def forward(self, images):
        """
        :param images: Batch containing several images to be processed.
        """
        images = torch.unsqueeze(images, 1)
        first_conv = self.first_conv(images)
        first_pooling = self.first_pooling(self.activation(first_conv))
        second_conv = self.second_conv(first_pooling)
        second_pooling = self.second_pooling(self.activation(second_conv))
        second_pooling = second_pooling.view(-1, 2*4*30)
        mlp_hidden_out = self.mlp_activation(self.mlp_l0_linear(second_pooling))
        return self.mlp_l1_linear(mlp_hidden_out)


class ConvNN3(nn.Module):
    """ This model is similar to the last one but with one more convolution layer before the MLP. """
    def __init__(self, input_shape, n_classes, mlp_hidden_units=256):
        """
        :param input_shape: Tuple containing the input matrix size.
        :param n_classes: Number of classes we want to classify.
        :param mlp_hidden_units: Number of hidden units in the MLP.
        :param dropout: Dropout to be applied before the softmax.
        """
        super(ConvNN3, self).__init__()
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.activation = nn.ReLU()
        self.first_conv = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, padding=2, bias=True)
        self.first_pooling = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=1)
        self.second_conv = nn.Conv2d(in_channels=20, out_channels=30, kernel_size=3, padding=2, bias=True)
        self.second_pooling = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.third_conv = nn.Conv2d(in_channels=30, out_channels=30, kernel_size=2, padding=1, stride=1, bias=True)
        self.third_pooling = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.mlp_l0_linear = nn.Linear(3*4*30, mlp_hidden_units)
        self.mlp_activation = nn.Sigmoid()
        self.mlp_l1_linear = nn.Linear(mlp_hidden_units, n_classes)

    def forward(self, images):
        """
        :param images: Batch containing several images to be processed.
        """
        images = torch.unsqueeze(images, 1)
        first_conv = self.first_conv(images)
        first_pooling = self.first_pooling(self.activation(first_conv))
        second_conv = self.second_conv(first_pooling)
        second_pooling = self.second_pooling(self.activation(second_conv))
        third_conv = self.third_conv(second_pooling)
        third_pooling = self.third_pooling(self.activation(third_conv))
        third_pooling = third_pooling.view(-1, 3*4*30)
        mlp_hidden_out = self.mlp_activation(self.mlp_l0_linear(third_pooling))
        return self.mlp_l1_linear(mlp_hidden_out)

# ----------------------------------------------------------------------------------------------------------------
#                                              Main                                                                 |
# ----------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Epochs to run.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate to be used.")
    parser.add_argument("--optim", type=str, default="Adam", help="Optimizer function name.")
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size to be used during training.")
    parser.add_argument("--cuda", type=bool, default=True, help="Flag to train the model in CUDA.")
    parser.add_argument("--model", type=str, default="ConvNN", help="Model class to be used.")
    parser.add_argument("--weight_decay", type=float, default=0., help="Optimizer weight decay parameter.")
    parser.add_argument("--filters", type=bool, default=False, help="Flag to print the 2 randomly selected filters.")
    args = parser.parse_args()
    
    # Load data.
    train_x, train_y = joblib.load("data/train.pkl")
    dev_x, dev_y = joblib.load("data/dev.pkl")
    test_x, test_y = joblib.load("data/test.pkl")
    train_data = utils.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    dev_data = utils.TensorDataset(torch.from_numpy(dev_x), torch.from_numpy(dev_y))
    test_data = utils.TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    train_loader = utils.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    dev_loader = utils.DataLoader(dev_data, batch_size=args.batch_size, shuffle=True)
    test_loader = utils.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    # Create model.
    if args.model == "ConvNN2":
        conv_nn = ConvNN2(train_x[0].shape, np.unique(train_y).shape[0])
    elif args.model == "ConvNN3":
        conv_nn = ConvNN3(train_x[0].shape, np.unique(train_y).shape[0])
    else:
        conv_nn = ConvNN1(train_x[0].shape, np.unique(train_y).shape[0])

    print ("Model: {}".format(conv_nn.__class__.__name__))
    # Train
    model, train_accuracy, dev_accuracy, losses = train(
        conv_nn, train_loader, dev_loader, optim=args.optim, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay
    )
    plot_loss(losses, "{}-losses.png".format(args.model))
    plot_train(train_accuracy, dev_accuracy, "{}-train-accuracy.png".format(args.model))
    print ("Test Accuracy: {0:.6f}".format(evaluate(model, test_loader)))

    if args.filters:
        model.print_filters()

if __name__ == '__main__':
    main()
