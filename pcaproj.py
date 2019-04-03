import numpy as np
def run(X, mu, Z):
    (n,d) = np.shape(X)
    for t in range(0,n):
        for i in range(0,d):
            X[t][i] = X[t][i] - mu[i]
    P = np.dot(X,Z)
    
    return P
