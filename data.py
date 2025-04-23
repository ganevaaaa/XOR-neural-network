import numpy as np


def get_data():
    #XOR inputs (X)
    X = np.array([[0, 0],
                  [1, 0],
                  [0, 1],
                  [1, 1]])


    # XOR outputs (Y)
    Y = np.array([[0],
                 [1],
                 [1],
                 [0]])
    return X, Y

