
import numpy as np
import pickle

d1 = np.loadtxt("datatest.txt",
    delimiter=',',skiprows=1,usecols=range(2,8))

d2 = np.loadtxt("datatest2.txt",
    delimiter=',',skiprows=1,usecols=range(2,8))

d3 = np.loadtxt("datatraining.txt",
    delimiter=',',skiprows=1,usecols=range(2,8))

dall = np.concatenate([d1, d2, d3])

np.random.shuffle(dall)

pos = int(len(dall) * 0.3)
pos += (len(dall) - pos) %5
testset = dall[:pos]
training = dall[pos:]

# k-folder value k
k = 5
# kf = [ [] for i in range(k)]
kf = np.split(training,5)
X = [ [] for i in range(k)]
Y = [ [] for i in range(k)]
for i in range(k):
    X[i] = kf[i][:,0:5]
    Y[i] = kf[i][:,5]
    # convert 0 to -1 for both training and testing dataset
    training_with_two = tuple(np.where(Y[i]==0))
    Y[i][training_with_two] = -1

XTest = testset[:,0:5]
yTest = testset[:,5]
testing_with_two = tuple(np.where(yTest==0))
yTest[testing_with_two] = -1
with open('testset.p','wb') as fl:
    pickle.dump([XTest,yTest],fl)
with open('kf.p','wb') as fl:
    pickle.dump([X,Y],fl)