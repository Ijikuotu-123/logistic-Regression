import numpy as np
import matplotlib.pyplot as plt

N =4
D=2

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1],
])

T = np.array([0,1,1,0])

ones = np.array([[1]*N]).T

#plt.plot(X[:,0],X[:,1])
#plt.show()

XY = np.matrix(X[:,0]* X[:,1]).T  # in the XOR problem and extra column is needed apart from the X
Xb = np.array(np.concatenate((ones,XY,X), axis =1))

# randomly initialize the weight
w = np.random.randn(D+2)

# calculate the model output
z = Xb.dot(w)

def sigmoid(z):
    return 1/ (1 + np.exp(-z))

Y = sigmoid(z)

# calculate the cross_entropy error
def cross_entropy(T,Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])            # np.log(Y[i]) gives a -ve value
        else:
            E -= np.log(1 -Y[i])
    return E

learning_rate = 0.001
error = []
for i in range(10000):
    e = cross_entropy(T, Y)
    error.append(e)
    if i % 1000 == 0:
        print (e)

    w += learning_rate * (np.dot((T-Y).T, Xb) -0.01*w)

    Y = sigmoid(Xb.dot(w))
plt.plot(error)
plt.plot("Cross_Entropy")

print ("final w:" , w)
print ("Final classification rate:", 1 -np.abs(T - np.round(Y)) .sum()/N)