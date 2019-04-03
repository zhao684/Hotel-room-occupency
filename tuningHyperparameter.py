import numpy as np
import matplotlib as plt
import pickle
#set backend for matplotlib
plt.use('Agg')
import matplotlib.pyplot as pp
#install sklearn first : using command "pip install -U scikit-learn"
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

#functions

def plotAccuracy(x,y,title,x_label,col,fileName):
	pp.figure()
	pp.plot(x, y, color = col)
	pp.title(title)
	pp.xlabel(x_label)
	pp.savefig(fileName, dpi=150)
	
def getSVMAccuracy(X_train,y_train, X_test, y_test,c):
	clf = svm.SVC(C=c,kernel = 'rbf', gamma = 'scale')
	clf.fit(X_train, y_train)
	(n, dx) = X_test.shape
	pred_set = []
	for j in range(n):
		pred = clf.predict(X_test[j].reshape((1,5)))
		pred_set.append(pred)
	accuracy = np.mean(np.array(pred_set).reshape((1,-1))==y_test)
	return accuracy
	
def getAdaBoostAccuracy(X_train,y_train, X_test, y_test, num):
	bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators = num)
	bdt.fit(X_train, y_train)
	(n, dx) = X_test.shape
	pred_set = []
	for j in range(n):
		pred = bdt.predict(X_test[j].reshape((1,5)));
		pred_set.append(pred)
	accuracy = np.mean(np.array(pred_set).reshape((1,-1))==y_test)
	return accuracy
def inter(X_train,y_train, X_test, y_test, num):
	return getAdaBoostAccuracy(X_train,y_train, X_test, y_test, num)

	
# main
k = 5

X,Y = pickle.load(open("kf.p","rb"))
Xtest,ytest = pickle.load(open("testset.p","rb"))

#select better kernel for SVM with fix C before K-fold

X_train = np.concatenate([x for xi,x in enumerate(X)])
y_train = np.concatenate([y for yi,y in enumerate(Y)])
clf = svm.SVC(C=1,kernel = 'rbf', gamma = 'scale')
clf.fit(X_train, y_train)
(n, dx) = Xtest.shape
pred_set = []
for i in range(n):
	pred = clf.predict(Xtest[i].reshape((1,5)))
	pred_set.append(pred)
accuracyRBF = np.mean(np.array(pred_set).reshape((1,-1))==ytest)

clf = svm.SVC(C=1,kernel = 'linear', gamma = 'scale')
clf.fit(X_train, y_train)
(n, dx) = Xtest.shape
pred_set = []
for i in range(n):
	pred = clf.predict(Xtest[i].reshape((1,5)))
	pred_set.append(pred)
accuracyLinear = np.mean(np.array(pred_set).reshape((1,-1))==ytest)

print accuracyRBF, accuracyLinear

#K-fold for selection of hyperparameter C in SVM
accuracy_set = []
C = []
accuracy_mean = []
accuracy_var = []

for c in np.arange(0.5, 2.5, 0.1):
	for i in range(k): 
		X_train = np.concatenate([x for xi,x in enumerate(X) if xi!= i])
		y_train = np.concatenate([y for yi,y in enumerate(Y) if yi!= i])
		X_test = np.concatenate([x for xi,x in enumerate(X) if xi== i])
		y_test = np.concatenate([y for yi,y in enumerate(Y) if yi== i])
		accuracy = getSVMAccuracy(X_train,y_train, X_test, y_test,c)
		accuracy_set.append(accuracy)
	accuracy_mean.append(np.mean(accuracy_set))
	accuracy_var.append(np.var(accuracy_set))
	C.append(c)

#plot (SVM)
plotAccuracy(C, accuracy_var, "SVM: Variance vs. C", "C [0.5, 2.5)", "red", "./Var (SVM)")
plotAccuracy(C, accuracy_mean, "SVM: Accuracy Mean vs. C", "C [0.5, 2.5)", "green", "./Mean (SVM)")



#K-fold for selection of number of weak classifier in AdaBoost
accuracy_set = []
WClassifier = []
accuracy_mean = []
accuracy_var = []

for w in range(90,131,5):
	for i in range(k):
		X_train = np.concatenate([x for xi,x in enumerate(X) if xi!= i])
		y_train = np.concatenate([y for yi,y in enumerate(Y) if yi!= i])
		X_test = np.concatenate([x for xi,x in enumerate(X) if xi== i])
		y_test = np.concatenate([y for yi,y in enumerate(Y) if yi== i])
		#accu = inter(X_train,y_train, X_test, y_test,w)
		accuracy_set.append(inter(X_train,y_train, X_test, y_test, w))
	accuracy_mean.append(np.mean(accuracy_set))
	accuracy_var.append(np.var(accuracy_set))
	WClassifier.append(w)

#plot accuracy (AdaBoost)
plotAccuracy(WClassifier, accuracy_var, "AdaBoost: Variance vs. # of weak classifier", "# of weak classifier [90,130]", "red", "./Var (AdaBoost)")
plotAccuracy(WClassifier, accuracy_mean, "AdaBoost: Accuracy Mean vs. # of weak classifier", "# of weak classifier [90,130]", "green", "./Mean (AdaBoost)")
