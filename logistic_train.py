import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import Shuffle
from process import get_binary_data

X, Y = get_binary_data()
X, Y = Shuffle(X, Y)

Xtrain = X[:-100]
Ytrain = X[:-100]
Xtest = X[-100:]
Ytest = X[-100:]

D = X.shape[1]   # this gives the dimension
w = np.random.randn(D)   # randomly initializing w
b = 0

def sigmoid(a):
    return 1/(1 + np.epx(-a))

def forward(X,w,b):
    return sigmoid(X.dot(w) + b)

def classification_rate(Y, P):
    return np.mean (Y ==P)

def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY) + (1-T)*np.log(1-pY))

train_costs = []
test_costs = []
learning_rate = 0.001
for i in range (10000):

    pYtrain = forward(Xtrain,w,b)
    pYtest = forward(Xtest,w,b)

    ctrain = cross_entropy(Ytrain, pYtrain)
    ctest = cross_entropy(Ytest, pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    w -= learning_rate*Xtrain.T.dot(pYtrain - Ytrain)
    b -= learning_rate*(pYtrain - Ytrain).sum()
    if i % 1000 == 0:
        print(i,ctrain, ctest)

print("final train classification_rate:", classification_rate(Ytain, np.round(pYtrain)))
print("final test classification_rate:", classification_rate(Ytest, np.round(pYtest)))