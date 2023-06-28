""" find w the probabilistic way. Assumption: data is from 2 classes and they are gaussian distributed,
same covariance but different mean. The best is to use gradient descent
"""

"""Bayers Rule
p(Y|X) = p(X|Y)p(Y)/p(X)"""

"""Cross-Entropy/ Error (j) or E = -{tlog(y) +(1-t)log(1-y)}   where t = target, y = logistic output
if t =1 only the first term matters, if t = 0 only second term matters"""

"""Multiple Samples
(j) = -summation(tnlog(yn) +(1-tn)log(1-yn))
        n=1
"""

import numpy as np

N = 100
D =2

X = np.random.randn(N,D)

# centers the first 50 points at (-2, -2)
X[:50, :] = X[:50, :]- 2*np.ones((50,D))

# centers the last 50 points at (2, 2)
X[50:, :] = X[50:, :] + 2*np.ones((50,D))

# labels: first 50 are 0 and last 50 are 1
T = np.array([0]*50 + [1]*50)
# add a column of ones
ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis = 1)

# randomly initialize the weights
w = np.random.randn(D+1)

# calculate the model output
z = Xb.dot(w)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

Y = sigmoid(z)

def cross_entropy(T,Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])            # np.log(Y[i]) gives a -ve value
        else:
            E -= np.log(1 -Y[i])
    return E

print (cross_entropy(T,Y))

# close_form solution
w = np.array([0,4,4])
z = Xb.dot(w)
Y = sigmoid(z)

print (cross_entropy(T,Y))


"""UPDATING WEIGHT USING GRADIENT DESCENT"""
# dj/dw = X.Transpose(Y -T)           
# w = w - learning_rate(dj/dw)

import numpy as np

N = 100
D =2

X = np.random.randn(N,D)

# centers the first 50 points at (-2, -2)
X[:50, :] = X[:50, :]- 2*np.ones((50,D))

# centers the last 50 points at (2, 2)
X[50:, :] = X[50:, :] + 2*np.ones((50,D))

# labels: first 50 are 0 and last 50 are 1
T = np.array([0]*50 + [1]*50)
# add a column of ones
ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis = 1) 
# with concantenate, i pick the axis where the concantenation should occur

# randomly initialize the weights
w = np.random.randn(D+1)

# calculate the model output
z = Xb.dot(w)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

Y = sigmoid(z)

def cross_entropy(T,Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])            # np.log(Y[i]) gives a -ve value
        else:
            E -= np.log(1 -Y[i])
    return E

#print (cross_entropy(T,Y))

learning_rate = 0.1
for i in range(100):
    if i% 10 ==0:
        print(cross_entropy(T, Y))

    w -= learning_rate * X.T.dot(T-Y)
    Y = sigmoid(Xb.dot(w))

#print( "final w:", w)



