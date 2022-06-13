from typing import *
import math
import h5py

import numpy as np
from numpy.linalg import norm
import random

import progetto.kfold as kfold


with h5py.File("data/usps.h5", 'r') as hf:
    train = hf.get('train')
    X_tr = train.get('data')[:]
    y_tr = train.get('target')[:]
    test = hf.get('test')
    X_te = test.get('data')[:]
    y_te = test.get('target')[:]

    all_X, all_Y = kfold.__merge_h5_train_test_sets(X_tr, y_tr, X_te, y_te)

    
def kernel_poly(x1, x2, exp=2):
    return (1 + x1.dot(x2)) ** exp

def kernel_gauss(x1, x2, gamma=0.25):
    return math.exp(-(norm(x1-x2)**2)/(2*gamma))
    
def sign(x):
    return 1 if x >= 0 else -1


# =========================================================== #
# =========================================================== #
# Preprocessing

# Data already normalized
assert max(np.max(X_tr), np.max(X_te)) == 1
assert min(np.min(X_tr), np.min(X_te)) == 0

# Meaningless values in labels? (just a sanity check)
for i, y in enumerate(y_tr):
    assert y in [0,1,2,3,4,5,6,7,8,9]

for i, y in enumerate(y_te):
    assert y in [0,1,2,3,4,5,6,7,8,9]


# =========================================================== #
# =========================================================== #
# Training algorithms

def pegasos(X, Y, for_digit, kernel, lambd, T) -> Callable:
    """
    Train a predictor for a given digit.
    """
    alpha, t = [], 1

    while (t < T):
        idx = random.randint(0, len(X)-1)
        x, y = X[idx], Y[idx]

        if t%1000 == 0:
            print(t, len(alpha))
            
        y = 1 if y == for_digit else -1

        prediction  = (1/(lambd*t))
        prediction *= sum(ys * kernel(xs, x) for xs,ys in alpha)

        # print(for_digit, Y[idx], prediction)
        
        # if sign(prediction) != y:
        #     alpha.append((x,y))
        # prediction  = y / (lambd*t)
        # prediction *= sum(ys * kernel(x, xs) for xs,ys in alpha)

        if y * prediction < 1:
            alpha.append((x,y))
            
            print(f"Wrong: {for_digit} {Y[idx]} {prediction}")
        else:
            pass
        
        t += 1

    print(t, len(alpha))
    
    return lambda x: (
        sum(ys * kernel(xs, x) for xs,ys in alpha)
    )


def pegasos_predictors(X, Y, lambd, T, kernel) -> List[Callable]:
    """
    Train predictors for all 10 digits.
    """
    predictors = list()
    
    for digit in range(0, 10):
        print(f"Building predictor for digit {digit}")
        
        predictor = pegasos(X,Y,digit,kernel,lambd,T)
        predictors.append(predictor)

    return predictors


# =========================================================== #
# =========================================================== #
# Test error calculation

def test_error(X, Y, predictors: List[Callable]) -> float:
    errors = 0

    print("Testing...")
    
    for t, (x,y) in enumerate(zip(X,Y)):
        if t!=0 and t%500 == 0:
            print(t)

        predictions = [predictor(x) for predictor in predictors]
        predictions = np.array(predictions)

        best_num = np.argmax(predictions)

        if best_num != y:
            errors += 1

    print(f"Test error: {errors/t}")
    return errors/t

# =========================================================== #
# =========================================================== #


def single_train_and_test(X, Y, lambd, T, kernel) -> Tuple[float, int, float]:
    """
    Performs an estimation of test error using 5-fold cross-validation.
    """
    print(f"Running on: lambda={lambd} and T={T}")
    
    test_errors = list()
    times       = list()
    
    for (xtr, ytr), (xte, yte) in kfold.generate_folds(X, Y, 5):
        start = time.time()

        # Build predictors for all digits
        predictors = pegasos_predictors(xtr, ytr, lambd, T, kernel)
        print(f"Predictors generation took: {int(time.time() - start)} seconds")
        
        test_errors.append(
            test_error(xte, yte, predictors)
        )
        
        times.append(int(time.time() - start))
        
        print(f"Single round test-error for lambda={lambd}, T={T}: " +
              f"{test_errors[-1]} in {times[-1]} seconds")

        
    print("Completed 5-fold train and test")
    print(f"Average test error: {sum(test_errors) / len(test_errors)}")
    print(f"Average duration of single round: {int(sum(times)/len(times))} seconds")
    print(f"Total time: {sum(times)} seconds")
    
    return (lambd, T, sum(test_errors) / len(test_errors))


# =========================================================== #
import time

__trainset_size = 4 * (len(all_X) // 5)
# Ts = [int(x * __trainset_size) for x in [0.1, 0.5, 1]]
Ts = [int(x * __trainset_size) for x in [0.5, 0.1, 0.5, 1]]

results = list()
last_time = time.time()

# for kernel in [kernel_gauss, kernel_poly, lambda x,y:kernel_poly(x,y,7)]:
    # kernel_time = time.time()
    
for T in Ts:
    for lambd in [10**-6, 10**-5, 10**-4, 10**-3]:
        lambd_time = time.time()

        results.append(
            single_train_and_test(all_X, all_Y, lambd, T,
                                  # lambda x,y:kernel_poly(x,y,7))
                                  kernel_gauss)
        )

        with open("results-gausskernel.csv", "a") as f:
            f.write(f"{lambd}, {T}, {results[-1][-1]}\n")

        print(f"Total time: {time.time() - lambd_time} seconds\n\n")

print(f"Total time: {time.time() - last_time}")
# =========================================================== #

