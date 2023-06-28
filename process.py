""" this is an E-Commerce project. it has six columns. the last column is the output, the 5th is the
 time column which will be divided into four columns(12am-6am =0, 6am-12pm=1, 12pm-6pm=2 and 6pm-12am=3)"""

import numpy as np
import pandas as pd

def get_data():
    df = pd.read_csv("ecommerce_data.csv")  # reading the csv file
    data = df.as_matrix()                   # changing the df to matrix

    X = data[ : , :-1]    # splitting data into X and Y
    Y = data[ : , -1]

    # numerical values (whose values can be large) that are not category should be normalized
    X[ :, 1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()   # normalizing duration of visit
    X[ :, 2] = (X[:,2 ]- X[:,2].mean()) / X[:,2].std()

    # Creating an array to accommodate the 3 additional columns for time and filling it with the old X
    N,D = X.shape
    X2 = np.zeros(N, D+3)
    X2[:,0:(D-1)] = X[:,0:(D-1)]

    # filling the remaining 4 columns of time . Method 1
    for n in range(N):
        t = int(X[n,D-1])
        X2[n,t+D-1] = 1

    # method 2
    Z = np.zeroes(N,4)
    Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1
    assert(nb.abs(X2[:,-4] - Z).sum() < 10e-10)

    return X2, Y

def get_binary_data():   # return x and y where the values of y <= 1
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2


