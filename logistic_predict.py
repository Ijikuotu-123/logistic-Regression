import numpy as np
from process import get_binary_data

X, Y = get_binary_data()

D = X.shape[1]
w = np.random.randn(D)
b = 0

def sigmoid(a):
    return 1 / ( 1 + np.exp(-a))

def forward(X, w,b):
    return sigmoid(X.dot(w) + b)

P_Y_given_X = forward (X,w,b)
predictions = np.round(P_Y_given_X)   # round off the value of y to 0 or 1

def classification_rate(Y,p):# it takes our targets and predictions. it divides the total correct over total number
    return np.mean(Y ==p)

print ("Score:", classification_rate(Y, predictions))
