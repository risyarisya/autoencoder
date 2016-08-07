import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

pi_k = np.array([0.3, 0.7])
x = np.linspace(0, 1, 10000)
y = np.linspace(0, 1, 10000)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 0.1, 0.2, 0.3, 0.5)
Z2 = mlab.bivariate_normal(X, Y, 0.1, 0.2, 0.7, 0.6)
Z = pi_k[0]*Z1 + pi_k[1]*Z2

#CS = plt.contour(X, Y, Z)
#plt.clabel(CS, inline=1, fontsize=10)
#plt.show() 

def mnd(_x, _mu, _sig):
    x = np.matrix(_x)
    mu = np.matrix(_mu)
    sig = np.matrix(_sig)
    a = np.sqrt(np.linalg.det(sig)*(2*np.pi)**sig.ndim)
    b = np.linalg.det(-0.5*(x-mu)*sig.I*(x-mu).T)
    return no.exp(b)/a
