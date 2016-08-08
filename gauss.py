import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

#pi_k = np.array([0.3, 0.7])
#x = np.linspace(0, 1, 10000)
#y = np.linspace(0, 1, 10000)
#X, Y = np.meshgrid(x, y)
#Z1 = mlab.bivariate_normal(X, Y, 0.1, 0.2, 0.3, 0.5)
#Z2 = mlab.bivariate_normal(X, Y, 0.1, 0.2, 0.7, 0.6)
#Z = pi_k[0]*Z1 + pi_k[1]*Z2

#CS = plt.contour(X, Y, Z)
#plt.clabel(CS, inline=1, fontsize=10)
#plt.show() 

import scipy.stats as stats


mu = np.array([0.5, 0.5])
sigma = np.array([[0.2,0],
                  [0,0.2]])

mu1 = np.array([0.3, 0.5])
sigma1 = np.array([[0.1, 0], [0, 0.2]])

mu2 = np.array([0.6, 0.7])
sigma2 = np.array([[0.2, 0], [0, 0.1]])

det = np.linalg.det(sigma)
det1 = np.linalg.det(sigma1)
det2 = np.linalg.det(sigma2)

inv_sigma = np.linalg.inv(sigma)
inv_sigma1 = np.linalg.inv(sigma1)
inv_sigma2 = np.linalg.inv(sigma2)
 
x = np.linspace(0, 1, 400)
y = np.linspace(0, 1, 400)
X, Y = np.meshgrid(x, y)
f = lambda x, y: stats.multivariate_normal(mu, sigma).pdf([x, y])
f2 = lambda x, y: stats.multivariate_normal(mu2, sigma2).pdf([x,y])
f1 = lambda x, y: stats.multivariate_normal(mu1, sigma1).pdf([x,y])

Z1 = np.vectorize(f1)(X, Y)
Z2 = np.vectorize(f2)(X, Y)
Z = 0.3*Z1+0.7*Z2

CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
print("hello!")
plt.show() 


