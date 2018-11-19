import numpy as np
import matplotlib.pyplot as plt


def read_data(filepath, partitions=None, pairwise_features=False):
    """Read the OCR dataset."""
    labels = {}
    f = open(filepath)
    X = []
    y = []
    for line in f:
        line = line.rstrip('\t\n')
        fields = line.split('\t')
        letter = fields[1]
        if letter in labels:
            k = labels[letter]
        else:
            k = len(labels)
            labels[letter] = k
        partition = int(fields[5])
        if partitions is not None and partition not in partitions:
            continue
        x = np.array([float(v) for v in fields[6:]])
        if pairwise_features:
            x = x[:, None].dot(x[None, :]).flatten()
        X.append(x)
        y.append(k)
    f.close()
    l = ['' for k in labels]
    for letter in labels:
        l[labels[letter]] = letter
    return X, y, l


def train_perceptron(X_train, y_train, X_dev, y_dev, labels, num_epochs=20):
    """Train with perceptron."""
    accuracies = []
    W = np.zeros((len(labels), len(X_train[0])))
    for epoch in range(num_epochs):
        num_mistakes = 0
        for j in range(len(y_train)):
            i = int(np.floor(len(y_train) * np.random.rand()))
            x = X_train[i]
            y = y_train[i]
            scores = W.dot(x)
            y_hat = np.argmax(scores)
            if y != y_hat:
                W[y] += x
                W[y_hat] -= x
                num_mistakes += 1
        accuracy = evaluate(W, X_dev, y_dev, labels)
        accuracies.append(accuracy)
        print('Epoch: %d, Mistakes: %d, Dev accuracy: %f' % (
                epoch+1, num_mistakes, accuracy))
    return W, accuracies


def train_logistic_sgd(X_train, y_train, X_dev, y_dev, labels, eta=0.1, reg=0.,
                       num_epochs=20):
    """Train logistic regression model with SGD."""
    accuracies = []
    W = np.zeros((len(labels), len(X_train[0])))
    for epoch in range(num_epochs):
        loss = 0.
        for j in range(len(y_train)):
            i = int(np.floor(len(y_train) * np.random.rand()))
            x = X_train[i]
            y = y_train[i]
            scores = W.dot(x)
            p = np.exp(scores) / np.sum(np.exp(scores))
            loss -= np.log(p[y]) / len(y_train)
            W *= (1. - reg*eta)
            W[y] += eta * x
            W -= eta * p[:, None].dot(x[None, :])
        accuracy = evaluate(W, X_dev, y_dev, labels)
        accuracies.append(accuracy)
        print('Epoch: %d, Loss: %f, Obj: %f, Dev accuracy: %f' % (
            epoch+1, loss, loss + 0.5*reg*(W*W).sum(), accuracy))
    return W, accuracies


def evaluate(W, X_test, y_test, labels):
    """Evaluate model on data."""
    correct = 0
    for i in range(len(y_test)):
        x = X_test[i]
        y = y_test[i]
        scores = W.dot(x)
        y_hat = np.argmax(scores)
        if y == y_hat:
            correct += 1
    return float(correct) / len(y_test)


def main():
    """Main function."""
    import argparse
    # Parse arguments.
    parser = argparse.ArgumentParser()

    parser.add_argument('filepath', type=str)
    parser.add_argument('--model', type=str, default='perceptron')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--eta', type=float, default=.001)
    parser.add_argument('--reg', type=float, default=0.)
    parser.add_argument('--pairwise', action='store_const', const=True,
                        default=False)

    args = vars(parser.parse_args())
    print(args)

    filepath = args['filepath']
    model = args['model']
    reg = args['reg']
    eta = args['eta']
    num_epochs = args['num_epochs']
    pairwise = args['pairwise']

    np.random.seed(42)

    print('Loading data...')
    X_train, y_train, labels = read_data(filepath, partitions=set(range(8)),
                                         pairwise_features=pairwise)
    X_dev, y_dev, _ = read_data(filepath, partitions={8},
                                pairwise_features=pairwise)
    X_test, y_test, _ = read_data(filepath, partitions={9},
                                  pairwise_features=pairwise)

    print('Training %s model...' % model)
    if model == 'perceptron':
        W, accuracies = train_perceptron(X_train, y_train, X_dev, y_dev, labels,
                                         num_epochs=num_epochs)
    elif model == 'logistic':
        W, accuracies = train_logistic_sgd(X_train, y_train, X_dev, y_dev,
                                           labels, eta=eta, reg=reg,
                                           num_epochs=num_epochs)
    else:
        raise NotImplementedError

    plt.plot(range(1, num_epochs+1), accuracies, 'bo-')
    plt.title('Dev accuracy')
    if model == 'perceptron':
        name = model
    else:
        name = '%s_%s_%s' % (model, reg, eta)
    if pairwise:
        name += '_pairwise'
    plt.savefig('%s.pdf' % name)

    print('Evaluating...')
    accuracy = evaluate(W, X_test, y_test, labels)
    print('Test accuracy: %f' % accuracy)


if __name__ == "__main__":
    main()
    