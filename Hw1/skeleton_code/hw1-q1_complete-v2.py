#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel): #falta adicionar o bias?
    def update_weight(self, x_i, y_i, **kwargs):
    
        eta=1
        y_hat = np.argmax(self.W.dot(x_i))
        if y_hat != y_i:
            self.W[y_i,:] += eta*x_i
            self.W[y_hat,:] -= eta*x_i

class LogisticRegression(LinearModel): #fala adicionar o bias?
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        eta=learning_rate
        label_scores = np.expand_dims(self.W.dot(x_i), axis = 1)

        # One-hot encode true label (num_labels x 1).
        y_one_hot = np.zeros((np.size(self.W, 0),1))
        y_one_hot[y_i] = 1

        # Softmax function
        # This gives the label probabilities according to the model (num_labels x 1).
        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))
        
        # SGD update. W is num_labels x num_features.
        self.W = self.W + eta * (y_one_hot - label_probabilities).dot(np.expand_dims(x_i, axis = 1).T)

        # Q1.1b


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        self.W1 = np.random.normal(0.1, 0.1,(hidden_size, n_features))
        self.W2 = np.random.normal(0.1, 0.1,(n_classes, hidden_size))
        self.weights = [self.W1, self.W2]
        self.hidden_size = hidden_size
        self.b1 = np.zeros(hidden_size)
        self.b2 = np.zeros(n_classes)
        self.bias = [self.b1, self.b2]

    def predict(self, X):
        predicted_labels = []
        for x in X:
            # Compute forward pass and get the class with the highest probability
            output, _ = self.forward(x)
            y_hat = np.argmax(output)
            predicted_labels.append(y_hat)
        predicted_labels = np.array(predicted_labels)
        return predicted_labels

    def forward(self, X):
        num_layers = len(self.weights)
        hiddens = []
        # compute hidden layers
        for i in range(num_layers):
                h = X if i == 0 else hiddens[i-1]
                z = self.weights[i].dot(h) + self.bias[i]
                if i < num_layers-1:  # Assuming the output layer has no activation.
                    z = np.maximum(0, z)  # Apply ReLU activation function
                    hiddens.append(z)
        output = z
        return output, hiddens

    def compute_loss(self, output, y):
        # compute loss
        probs = np.exp(output - np.max(output)) / np.sum(np.exp(output - np.max(output)))
        loss = -y.dot(np.log(probs + 1e-5))
        
        return loss

    def backward(self, x, y_oneh, output, hiddens):
        #print ("y:", y_oneh)
        num_layers = len(self.weights)
        z = output
        probs = np.exp(output - np.max(output)) / np.sum(np.exp(output - np.max(output)))
        grad_z = probs - y_oneh 
        
        grad_weights = []
        grad_biases = []
        
        # Backpropagate gradient computations 
        for i in range(num_layers-1, -1, -1):
            
            # Gradient of hidden parameters.
            h = x if i == 0 else hiddens[i-1]
            grad_weights.append(grad_z[:, None].dot(h[:, None].T))
            grad_biases.append(grad_z)
            
            # Gradient of hidden layer below.
            grad_h = self.weights[i].T.dot(grad_z)

            # Gradient of hidden layer below before activation.
            grad_z = grad_h * (h > 0)   # Grad of loss

        # Making gradient vectors have the correct order
        grad_weights.reverse()
        grad_biases.reverse()
        return grad_weights, grad_biases


    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


    def train_epoch(self, X, y_train, learning_rate=0.001):
        n_classes = len(np.unique(y_train))
        num_layers = len(self.weights)
        total_loss = 0
        one_hot = np.zeros((np.size(y_train, 0), n_classes))
        for i in range(np.size(y_train, 0)):
            one_hot[i, y_train[i]] = 1
        y_train_ohe = one_hot
        for x, y in zip(X, y_train_ohe):
            # Compute forward pass
            output, hiddens = self.forward(x)
            
            # Compute Loss and Update total loss
            loss = self.compute_loss(output, y)
            total_loss+=loss
            # Compute backpropagation
            #print (y_oneh)
            grad_weights, grad_biases = self.backward(x, y, output, hiddens)
            
            # Update weights
            
            for i in range(num_layers):
                self.weights[i] -= learning_rate*grad_weights[i]
                self.bias[i] -= learning_rate*grad_biases[i]
                
        return total_loss



def plot(epochs, train_accs, val_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.show()

def plot_loss(epochs, loss):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs)
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss)


if __name__ == '__main__':
    main()
