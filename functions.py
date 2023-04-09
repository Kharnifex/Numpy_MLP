from tensorflow import keras
import numpy as np


def split_train_data(array):
    """
    Splits an array into two, the first one containing 20% of the original array's elements
    and the other containing the remaining 80%
    :param array: the array to be split
    :return: the two arrays
    """
    array_split = np.array_split(array, 5)

    array1 = array_split[0]
    array2 = np.concatenate((array_split[1], array_split[2], array_split[3], array_split[4]))

    return array1, array2


def sigmoid(z):
    """
    Calculates the value of the sigmoid function for number/vector/array z
    :return: the value of the sigmoid function
    """
    return 1.0 / (1.0 + np.exp(-z))


def loaddata():
    """
    Loads the MNIST dataset, turns each image into a vector of 28*28=784 pixels,
    changes the range of values for each pixel from int(0,255) to f(0,1)
    :return: three tuples of two arrays each, containing train/validation/test data
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train / 255
    x_test = x_test / 255

    train_filter = np.where((y_train == 5) | (y_train == 6))
    test_filter = np.where((y_test == 5) | (y_test == 6))

    y_train[y_train == 5] = 0
    y_test[y_test == 5] = 0
    y_train[y_train == 6] = 1
    y_test[y_test == 6] = 1

    x_train, y_train = x_train[train_filter], y_train[train_filter]
    x_test, y_test = x_test[test_filter], y_test[test_filter]

    x_train_flattened = x_train.reshape(len(x_train), 28 * 28)
    x_test_flattened = x_test.reshape(len(x_test), 28 * 28)

    x_valid, x_train_final = split_train_data(x_train_flattened)
    y_valid, y_train_final = split_train_data(y_train)

    return (x_train_final, y_train_final), (x_valid, y_valid), (x_test_flattened, y_test)


def load_and_prepare_data():
    """
    Loads the MNIST dataset using the loaddata() function,
    turns vectors of (x,) shape into arrays with (x,1) shape
    and adds a 1 in at the start of each x array
    :return: three tuples of two arrays each, containing train/validation/test data
    """
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = loaddata()

    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    y_valid = y_valid.reshape((-1, 1))

    # x_train = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
    # x_valid = np.hstack([np.ones((x_valid.shape[0], 1)), x_valid])
    # x_test = np.hstack([np.ones((x_test.shape[0], 1)), x_test])
    # above 3 lines could be useful for logistic regression,
    # however results are pretty much the same with and without them

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def stop_Early(listt, tol=1e-5):
    """
    A function that checks the last 5 elements of a list;  used only for early stopping
    :param tol: the tolerance of the model
    :param listt: the list of values (errors)
    :return: True if the difference between the last and 5th from last element of list is less than 0.001
    """
    if len(listt) < 40:  # don't stop for first 40 epochs
        return False
    else:
        x = listt[-5:]
        if x[0] - x[4] <= tol:
            return True
        else:
            return False
