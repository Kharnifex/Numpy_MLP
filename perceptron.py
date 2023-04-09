from functions import *
import numpy as np


class NeuralNetwork:
    """
    A class whose instances represent a neural network.
    All of its attributes are explained in the constructor's documentation.
    Author: Ioannis Gkionis
    """

    def __init__(self, hid, epochs, learning_rate, tol=1e-5, name='Neural Network Classifier'):
        """
        Constructs an instance of NeuralNetwork class

        NOTE 1: the numbers of neurons in the input and output layers are fixed (784 and 1 respectively),
                if this code is used for anything other than MNIST dataset binary classification
                they need to be changed or added as parameters to the constructor

        NOTE 2: some of the arrays/vectors explained below change shapes based on the shape of the input data.
                Wherever X exists in the declaration of an array's shape it refers to the amount of input data

        NOTE 3: some of the arrays/vectors(represented as matrices) declared and explained later might be
                transposed versions of what they actually represent. Pay close attention to how they're used.

        :param name: The name of the network (useful for managing multiple models
        :param hid: Number of neurons in the hidden layer
        :param tol: Tolerance - used for early stopping, (higher number -> model stops training earlier)
        :param epochs: Number of epochs in training
        :param learning_rate: The rate with which weights change

        The rest of the class's attributes are explained below

        Initialized with constructor call:

        n_in: Number of neurons in the input layer
        n_out: Number of neurons in the output layer
        losses: List of loss values for each epoch (can be used to plot the loss function)
        W_hid: Weight array for the hidden layer [SHAPE: (n_hid, n_in)]
        b_hid: Bias vector for the hidden layer [SHAPE: (n_hid,1)]
        W_out: Weight array for the output layer [SHAPE: (n_out, n_hid)]
        b_out: Bias vector for the output layer [SHAPE: (n_out,1)]

        Declared in forward pass:

        Z_hid: Weighted input array for the hidden layer [SHAPE: (n_hid, X)]
        A_hid: Activation array for the hidden layer [SHAPE: (n_hid, X)]
        Z_out: Weighted input array for the output layer [SHAPE: (n_out, X)]
        A_out: Activation array for the output layer [SHAPE: (n_out, X)]

        Partial Derivatives:

        dZ_out: Partial Derivative of matrix Z_out [SHAPE: same as Z_out]
        dW_out: Partial Derivative of matrix W_out [SHAPE: same as W_out]
        db_out: Partial Derivative of vector b_out [SHAPE: same as b_out]
        dZ_hid: Partial Derivative of matrix Z_hid [SHAPE: same as Z_hid]
        dW_hid: Partial Derivative of matrix W_hid [SHAPE: same as W_hid]
        db_hid: Partial Derivative of vector b_hid [SHAPE: same as b_hid]
        """
        self.n_in = 784
        self.n_hid = hid
        self.n_out = 1
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.tol = tol
        self.name = name

        self.losses = []

        self.W_hid = np.random.randn(self.n_hid, self.n_in) * 0.01  # initialize weights and biases
        self.b_hid = np.zeros((self.n_hid, 1))
        self.W_out = np.random.randn(self.n_out, self.n_hid) * 0.01
        self.b_out = np.zeros((self.n_out, 1))

    def forward(self, X):
        """
        Does a forward pass through the 3 layers of the network for a certain set of input data.
        In case the code gets confusing, revert to the documentation of the constructor
        :param X: the X array of the dataset
        """
        self.Z_hid = self.W_hid.dot(X.T) + self.b_hid
        self.A_hid = sigmoid(self.Z_hid)
        self.Z_out = self.W_out.dot(self.A_hid) + self.b_out
        self.A_out = sigmoid(self.Z_out)

    def back_prop(self, X, Y):
        """
        Propagates the gradient of the loss backwards through the network
        In case the code gets confusing, revert to the documentation of the constructor
        :param X: The X array of the set
        :param Y: The Y array of the set
        """
        self.dZ_out = self.A_out - Y.T  # (A_out-Y.T).dot(X) = Binary Cross-Entropy gradient
        self.dW_out = (1 / X.shape[0]) * np.dot(self.dZ_out, self.A_hid.T)
        self.db_out = (1 / X.shape[0]) * np.sum(self.dZ_out, axis=1, keepdims=True)
        self.dZ_hid = np.multiply(np.dot(self.W_out.T, self.dZ_out), 1 - np.power(self.A_hid, 2))
        self.dW_hid = (1 / X.shape[0]) * np.dot(self.dZ_hid, X)
        self.db_hid = (1 / X.shape[0]) * np.sum(self.dZ_hid, axis=1, keepdims=True)

    def train(self, X_train, Y_train, X_valid, Y_valid):
        """
        Trains the Neural Network for a certain X_train,Y_train set of data
        and validates it using a X_valid,Y_valid set
        Note: parameters such as epochs, learning rate etc are all set in the constructor, not here.
        :param X_train: the X array of the training set
        :param Y_train: the Y array of the training set
        :param X_valid: the X array of the validation set
        :param Y_valid: the Y array of the validation set
        :return the number of the epoch when training stopped
        """
        for e in range(1, self.epochs + 1):

            self.forward(X_valid)  # forward pass
            loss, grad = self.calculate_loss(X_valid, Y_valid)  # calculate and save loss for validation set
            self.losses.append(loss)

            if stop_Early(self.losses, self.tol):
                break

            self.forward(X_train)

            self.back_prop(X_train, Y_train)  # back propagate error

            self.W_hid -= self.learning_rate * self.dW_hid  # change weight/bias values
            self.b_hid -= self.learning_rate * self.db_hid
            self.W_out -= self.learning_rate * self.dW_out
            self.b_out -= self.learning_rate * self.db_out

        return e


    def predict(self, X):
        """
        Uses the model's trained weights to predict the Y vector for the X dataset
        :param X: The X array of the dataset
        :return: an array (NOT vector) of 0 and 1 values.
                 Note: the array needs to be transposed before checking whether the values
                 of the predictions are the same as the y array of the dataset
        """
        # Note: instead of this entire function we can just write
        # return np.round(sigmoid(sigmoid(X.dot(self.W_hid.T)).dot(self.W_out.T))).astype('int')
        # however this line of code is a bit too schizophrenic and we can make use of the forward() method instead
        self.forward(X)
        return np.round(self.A_out).astype('int')

    def show_predictions(self, x, y):
        """
        Calculates accuracy of current model for a x,y dataset
        :param x: the x array of the set
        :param y: the y array of the set
        :return: prints the value of the accuracy
        """
        z = self.predict(x).T  # check documentation of predict() in case the transpose part doesn't make sense
        print('Model: ' + self.name + ' Accuracy: ', np.mean(z.astype('int') == y))

    def calculate_loss(self, x, y):
        """
        Calculates the loss of the current model for validation set x, y using the Binary Cross-Entropy formula
        :param x: the x array of the set
        :param y: the y array of the set
        :return: A number ranging from 0 to 1 that indicates how well the model performs.
                 The better models score closer to 0 while worse ones closer to 1
        """
        cost = -np.sum(np.multiply(np.log(self.A_out.T), y) + np.multiply(np.log(1 - self.A_out.T), (1 - y))) / x.shape[
            0]
        grad = (self.A_out - y.T).T * x
        return cost, grad

    def gradcheck_softmax(self, X, Y):
        """
        Copied and adjusted function from uni lab as a method for NeuralNetwork class
        """
        epsilon = 1e-6

        _list = np.random.randint(X.shape[0], size=5)
        x_sample = np.array(X[_list, :])
        y_sample = np.array(Y[_list, :])

        self.forward(x_sample)
        Ew, gradEw = self.calculate_loss(x_sample, y_sample)
        numericalGrad = np.zeros(gradEw.shape)
        # Compute all numerical gradient estimates and store them in
        # the matrix numericalGrad
        for k in range(numericalGrad.shape[0]):
            for d in range(numericalGrad.shape[1]):
                # add epsilon to the w[k,d]
                self.W_hid[k, d] += epsilon
                e_plus, _ = self.calculate_loss(x_sample, y_sample)

                # subtract epsilon to the w[k,d]
                self.W_hid[k, d] -= epsilon
                e_minus, _ = self.calculate_loss(x_sample, y_sample)

                # approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
                numericalGrad[k, d] = (e_plus - e_minus) / (2 * epsilon)

        return gradEw, numericalGrad

    """def train_Stochastic(self, B, X_train, Y_train, X_valid, Y_valid):
        for e in range(1, self.epochs + 1):
            m = X_train.shape[0]
            z = m % B + 1

            self.forward(X_valid)  # forward pass
            loss, grad = self.calculate_loss(X_valid, Y_valid)  # calculate and save loss for validation set
            self.losses.append(loss)

            if stop_Early(self.losses, self.tol):
                print('Model ' + self.name + ' stopped early at ' + str(e) + ' epoch')
                break

            for j in range(0,z):
                x_batches = np.random.randint(X_train.shape[0], size=B)
                y_batches = np.random.randint(Y_train.shape[0], size=B)

                if len(X_train) > B:
                    for i in range(0, B - 1):
                        x_batches[i] = X_train[0]
                        y_batches[i] = Y_train[0]
                        np.delete(X_train, 0, 0)
                        np.delete(Y_train, 0, 0)

                self.forward(x_batches)

                self.back_prop(x_batches, y_batches)  # back propagate error

                self.W_hid -= self.learning_rate * self.dW_hid  # change weight/bias values
                self.b_hid -= self.learning_rate * self.db_hid
                self.W_out -= self.learning_rate * self.dW_out
                self.b_out -= self.learning_rate * self.db_out

            if e % 50 == 0:
                print("Validation Loss ", e, " = ", loss) # print loss and accuracy every 50 epochs
                self.show_predictions(X_train, Y_train)"""
