
# coding: utf-8

# In[2]:

#Aradhya Gupta, Ashwin Venkatesha


# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

data=np.loadtxt('clusters.txt',delimiter=',')
# fit a Gaussian Mixture Model with two components
clf = mixture.GaussianMixture(n_components=3, covariance_type='full', max_iter = 100000, init_params = 'kmeans')
clf.fit(data)
print("Amplitude :\n\n", clf.weights_)
print("nMeans :\n\n", clf.means_)
print("nCov :\n\n", clf.covariances_)

x = np.linspace(-6., 12.)
y = np.linspace(-6., 12.)

X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(data[:, 0], data[:, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()

