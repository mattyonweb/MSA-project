from typing import *
import math
import h5py
from numpy.linalg import norm

with h5py.File("../data/usps.h5", 'r') as hf:
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

def kernel_perceptron(X, Y, for_digit, kernel):
    errors = list()

    for i, (x, y) in enumerate(zip(X, Y)):
        if i % 500 == 0:
            print(i)
            
        y = 1 if y == for_digit else -1

        y_tilde = sign(sum( ys*kernel(x,xs) for xs,ys in errors))

        if y_tilde != y:
            errors.append((x,y))

    # return sum(y*x for x,y in errors)
    return lambda x: sum(ys*kernel(x, xs) for xs,ys in errors)


def test_error(X, Y, ws):
    errors, total = 0, len(Y)
    
    for x,y in zip(X,Y):        
        results = np.array([w(x) for w in ws])

        y_tilde = np.argmax(results)

        if y_tilde != y:
            errors += 1

    return errors/total
