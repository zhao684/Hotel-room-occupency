
# coding: utf-8

# In[3]:

import pcalearn
import pcaproj
import numpy as np
import matplotlib as plt
plt.use('Agg')
import pickle


# Just pca on first set in kfold

# In[4]:


Xl,Yl = pickle.load(open("kf.p","rb"))
X = Xl[0]
y = Yl[0]
F = 2
mu,Z = pcalearn.run(F, X)


# In[6]:


# print mu.shape
# print Z.shape


# In[7]:


X_small = pcaproj.run(X,mu,Z)


# In[11]:


# X_small.shape


# In[9]:


import matplotlib.pyplot as pp


# In[12]:


positive_samples = list(np.where(y==1)[0])
negative_samples = list(np.where(y==-1)[0])


# In[14]:


print len(positive_samples), len(negative_samples)


# In[17]:


pp.figure()
pp.plot(X_small[positive_samples,0], X_small[positive_samples,1], 'bo') # b=blue, o=circle
pp.plot(X_small[negative_samples,0], X_small[negative_samples,1], 'ro') # r=red, o=circle
pp.xlabel('PCA feature 0')
pp.ylabel('PCA feature 1')
pp.savefig("./PCA visual", dpi=150) 