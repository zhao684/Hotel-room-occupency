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

X,Y = pickle.load(open("kf.p","rb"))
Xtest,ytest = pickle.load(open("testset.p","rb"))
X_train = np.concatenate([x for xi,x in enumerate(X)])
y_train = np.concatenate([y for yi,y in enumerate(Y)])
#use ROC to select model

#ROC for SVM 
clf = svm.SVC(C=1.5,kernel = 'rbf', gamma = 'scale',probability=True)
clf.fit(X_train, y_train)
(n, dx) = Xtest.shape
pred_set = []
for i in range(n):
	pred = clf.predict_proba(Xtest[i].reshape((1,5)))
	pred_set.append(pred)
(neg,pos) = np.hsplit(np.array(pred_set).reshape((-1,2)),2)
oc_test = pos[tuple(np.where(ytest==1))]
noc_test = pos[tuple(np.where(ytest==-1))]
senSVM = []
speSVM = []

pos.sort(axis=0)
for threadhold in pos:
	temp_oc = pos[tuple(np.where(oc_test >= threadhold))]
	temp_noc = pos[tuple(np.where(noc_test < threadhold))]
	speSVM.append(1.0 * len(temp_oc)/len(oc_test))
	senSVM.append(1.0 * len(temp_noc)/len(noc_test))

#ROC for AdaBoost 
	
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators = 110)
bdt.fit(X_train, y_train)
(n, dx) = Xtest.shape
pred_set = []
for i in range(n):
	pred = bdt.predict_proba(Xtest[i].reshape((1,5)));
	pred_set.append(pred)
(neg,pos) = np.hsplit(np.array(pred_set).reshape((-1,2)),2)
oc_test = pos[tuple(np.where(ytest==1))]
noc_test = pos[tuple(np.where(ytest==-1))]
senAda = []
speAda = []
pos.sort(axis=0)
for threadhold in pos:
	temp_oc = pos[tuple(np.where(oc_test >= threadhold))]
	temp_noc = pos[tuple(np.where(noc_test < threadhold))]
	speAda.append(1.0 * len(temp_oc)/len(oc_test))
	senAda.append(1.0 * len(temp_noc)/len(noc_test))

pp.figure()
pp.xlim([0.0, 1.0])
pp.plot(speSVM, senSVM, color = 'r', linewidth=0.6)
pp.plot(speAda, senAda, color = 'g', linewidth=0.6)
pp.legend(('SVM (rbf, C=1.5)','AdaBoost(# of WC = 110)'),  loc = 'lower left')
pp.xlabel("Specificity rate")
pp.ylabel("Sensitivity rate")
pp.title("ROC graph for SVM and AdaBoost")
pp.savefig("./ROC (SVM)", dpi=150)

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators = 110)
bdt.fit(X_train, y_train)
(n, dx) = Xtest.shape
pred_set = []
for j in range(n):
	pred = bdt.predict(Xtest[j].reshape((1,5)));
	pred_set.append(pred)
accuracy = np.mean(np.array(pred_set).reshape((1,-1))==ytest)
print accuracy