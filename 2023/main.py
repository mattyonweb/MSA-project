from typing import *
import math
import h5py
from numpy.linalg import norm

with h5py.File("data/usps.h5", 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]
    
    
def kernel_poly(x1, x2, exp=2):
    return (1 + x1.T.dot(x2)) ** exp

def kernel_gauss(x1, x2, gamma=0.25):
    return math.exp(-(norm(x1-x2)**2)/(2*gamma))
    
def sign(x):
    return 1 if x >= 0 else -1

# =========================================================== #

def pegasos(X, Y, for_digit, kernel, lambd):
    alpha = []

    for t, (x,y) in enumerate(zip(X,Y)):
        if t%500 == 0:
            print(t, len(alpha))
            
        y = 1 if y == for_digit else -1
        
        prediction  = y / (lambd*(1+t))
        prediction *= sum(ys + kernel(x, xs) for xs,ys in alpha)

        if prediction <= 1:
            alpha.append((x,y))

    return lambda x: (
        sum(ys * kernel(xs, x) for xs,ys in alpha)
    )

predictors = list()       
for digit in range(0, 10):
    print(digit)
    predictor = pegasos(X_tr,y_tr,digit,kernel_gauss,10**-6)
    predictors.append(predictor)


import numpy as np

def testerror(X,Y,predictors):
        errors = 0
        
        for t, (x,y) in enumerate(zip(X,Y)):
                if t%500 == 0:
                        print(t)

                predictions = [predictor(x) for predictor in predictors]
                predictions = np.array(predictions)

                best_num = np.argmax(predictions)
                
                if best_num != y:
                        errors += 1

        print(errors/t)


testerror(X_te, y_te, predictors)

# =========================================================== #

import matplotlib.pyplot as plt

