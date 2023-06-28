import numpy as np

N = 100
D =2

X = np.random.randn(N,D)

# centers the first 50 points at (-2, -2)
X[:50, :] = X[:50, :] - 2*np.ones((50,D))

# centers the last 50 points at (2, 2)
X[50:, :] = X[50:, :] + 2*np.ones((50,D))

# labels: first 50 are 0 and last 50 are 1
t = np.array([0]*50 + [1]*50)
# add a column of ones for the bias
ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis = 1)

# randomly initialize the weights
w = np.random.randn(D+1)

# calculate the model output
z = Xb.dot(w)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

Y = sigmoid(z)

def cross_entropy(t,Y):
    E = 0
    for i in range(N):
        if t[i] == 1:
            E -= np.log(Y[i])            # np.log(Y[i]) gives a -ve value
        else:
            E -= np.log(1 -Y[i])
    return E

print (cross_entropy(t,Y))

learning_rate = 0.1
for i in range(100):
    if i% 10 ==0:
        print(cross_entropy(t, Y))

    w += learning_rate * (np.dot((t - Y).T, Xb) -0.1*w)
    Y = sigmoid(Xb.dot(w))
    

print( "final w:", w)
