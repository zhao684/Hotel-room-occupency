import numpy as np
import numpy.linalg as la
def run(F,X):
    (n,d) = np.shape(X)
    mu = np.zeros((d,1))
    for i in range(0,d):
        sum = 0
        for t in range(0,n):
            sum = sum + X[t][i]
        mu[i] = sum / n
    for t in range(0,n):
        for i in range (0,d):
            X[t][i] = X[t][i] - mu[i]
    U, s, Vt = la.svd(X,False)
    g = np.zeros((F,1))
    for i in range (0, F):
        g[i] = s[i]

    for i in range(0,F):
        if(g[i] > 0):
            g[i] = 1 / g[i]
    W = Vt[0:F,:]
    gg = []
    for i in range(0,len(g)):

        gg.append(g[i][0])

    g = np.array(gg)
    Z = np.dot(W.T, np.diag(g))
    return (mu,Z)
