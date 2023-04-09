from perceptron import *

(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_and_prepare_data()  # load all data once

def run_MLP(hid=64, ep=1000, lr=0.1, tol=0.001, name='Neural Network Classifier'):
    """
    A function that creates and trains a Neural Network Classifier
    :param tol: The tolerance used for early stopping
    :param hid: The amount of neurons in the hidden layer
    :param ep: The amount of epochs
    :param lr: The learning rate
    :param name: The name of the model
    """
    nn = NeuralNetwork(hid=hid, epochs=ep, learning_rate=lr, tol=tol, name=name)
    x = nn.train(X_train, y_train, X_valid, y_valid)

    nn.show_predictions(X_test, y_test)
    print('Model: ', name, ' stopped training at epoch number ', x)

    return