from typing import *
import numpy as np

def __merge_h5_train_test_sets(xtr: np.array, ytr, xte, yte):
    return (
        np.concatenate((xtr, xte)),
        np.concatenate((ytr, yte))
    )


def randomize_dataset(X: np.array, Y: np.array) -> Tuple[np.array, np.array]:
    permutation = np.random.permutation(len(X_tr))
    
    newx = X[permutation]
    newy = Y[permutation]
    
    return newx, newy


def generate_folds(X, Y, k):
    X, Y = randomize_dataset(X, Y)

    train_size, test_size = 4 * (len(x) // 5), len(x) // 5
    
    for i in range(k):
        testX, trainX = X[:test_size], X[test_size]
        testY, trainY = Y[:test_size], Y[test_size]
        
        yield (trainX, trainY), (testX, testY)

        X, Y = np.roll(X, k//5), np.roll(Y, k//5)
        
