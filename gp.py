from __future__ import division
import numpy as np
import matplotlib.pyplot as pl

""" This is code for simple GP regression. It assumes a zero mean GP Prior """


# This is the true unknown function we are trying to approximate
f = lambda x: np.sin(0.9*x).flatten()
#f = lambda x: (0.25*(x**2)).flatten()


# Define the kernel
def kernel(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 0.1
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)

N = 10         # number of training points.
n = 50         # number of test points.
s = 0.00005    # noise variance.

# Sample some input points and noisy versions of the function evaluated at
# these points. 
X = np.random.uniform(-5, 5, size=(N,1))
y = f(X) + s*np.random.randn(N)

K = kernel(X, X)
K_y = K + s*np.eye(N)
L = np.linalg.cholesky(K_y)

# points we're going to make predictions at.
Xtest = np.linspace(-5, 5, n).reshape(-1,1)

# compute the variance at our test points.
K_11 = kernel(Xtest, Xtest)
# draw samples from the prior at our test points.
L_1_prior = np.linalg.cholesky(K_11 + 1e-6*np.eye(n))
f_prior = np.dot(L_1_prior, np.random.normal(size=(n,10)))

K_1 = kernel(X, Xtest)
# compute the mean at our test points.
Lk = np.linalg.solve(L, K_1)  #L-1 K_*

m = np.linalg.solve(L, y)   #L-T L-1y
mu_1 = np.dot(Lk.T, m)  # K_*T K-1 y = (K_*T L-T) (L-1y) = Lk m

#K_*T K-1 K_* = (K_*T L-T) (L-1 K_*) = LkT Lk
s2 = np.diag(K_11) - np.sum(Lk**2, axis=0)
s = np.sqrt(s2)

L_1_post = np.linalg.cholesky(K_11 + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu_1.reshape(-1,1) + np.dot(L_1_post, np.random.normal(size=(n,10)))


# PLOTS:
pl.figure(1)
pl.clf()
pl.plot(X, y, 'r+', ms=20)
pl.plot(Xtest, f(Xtest), 'b-')
pl.gca().fill_between(Xtest.flat, mu_1-3*s, mu_1+3*s, color="#dddddd")
pl.plot(Xtest, mu_1, 'r--', lw=2)
pl.savefig('predictive.png', bbox_inches='tight')
pl.title('Mean predictions plus 3 st.deviations')
pl.axis([-5, 5, -3, 3])


pl.figure(2)
pl.clf()
pl.plot(Xtest, f_prior)
pl.title('Ten samples from the GP prior')
pl.axis([-5, 5, -3, 3])
pl.savefig('prior.png', bbox_inches='tight')

# draw samples from the posterior at our test points.

pl.figure(3)
pl.clf()
pl.plot(Xtest, f_post)
pl.title('Ten samples from the GP posterior')
pl.axis([-5, 5, -3, 3])
pl.savefig('post.png', bbox_inches='tight')

pl.show()
