# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 20:48:05 2020

@author: Nagnanamus
"""

from scipy.optimize import minimize
import scipy
import cvxpy as cvx
import cvxopt as cvxopt
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon, LineString
from shapely.geometry import MultiLineString
import matlab.engine
eng = matlab.engine.start_matlab()

np.random.seed(1)

# %%


def OptTrendFit_cvx(y, vlambda, weights):

    n = y.size
    # Form second difference matrix.
    e = np.mat(np.ones((1, n)))
    D = scipy.sparse.spdiags(np.vstack((e, -2 * e, e)), range(3), n - 2, n)
    # Convert D to cvxopt sparse format, due to bug in scipy which prevents
    # overloading neccessary for CVXPY. Use COOrdinate format as intermediate.
    # D_coo = D.tocoo()
    # D = cvxopt.spmatrix(D_coo.data, D_coo.row.tolist(), D_coo.col.tolist())

    # Set regularization parameter.

    # Solve l1 trend filtering problem.
    x = cvx.Variable(n)
    obj = cvx.Minimize(0.5 * cvx.sum_squares(cvx.multiply(weights, y - x))
                       + vlambda * cvx.norm(D * x, 1))
    prob = cvx.Problem(obj)
    # ECOS and SCS solvers fail to converge before
    # the iteration limit. Use CVXOPT instead.
    prob.solve(solver=cvx.CVXOPT, verbose=False)

    # print 'Solver status: ', prob.status
    # Check for error.
    if prob.status != cvx.OPTIMAL:
        raise Exception("Solver did not converge!")

    return np.array(x.value)


# %%
plt.close('all')

data = np.load('map1.npz')
truemap = data['truemap']

# robd=np.array([[0,1],[1,-1],[-1,-1],[0,1]])
robd = np.array([[2, 0], [-0.5, 0.5], [-0.5, -0.5], [2, 0]])

robrng = 300

# state is x,y,th
x0 = np.array([10, 10, np.pi / 2])


def plotrobot(x, ax):
    th = x[2]
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    ss = np.zeros(robd.shape)
    for i in range(len(robd)):
        ss[i, :] = np.matmul(R, robd[i, :]) + x[0:2]
    ax.plot(ss[:, 0], ss[:, 1], 'k')


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([-10, 100])
ax.set_ylim([-10, 100])

ax.plot(truemap[:, 0], truemap[:, 1], 'bo-')
plotrobot(x0, ax)
plt.show()


def recordmeas(truemap, x0, robrng):
    measbeam = np.array([x0[0:2], (500 * np.cos(x0[2]), 500 * np.sin(x0[2]))])
    poly1 = eng.polyshape(matlab.double(
        truemap[:, 0].tolist()), matlab.double(truemap[:, 1].tolist()))
    X = eng.intersect(poly1, matlab.double(measbeam.tolist()))
    #X = eng.polyxpoly(measbeam[:,0].tolist(),measbeam[:,1].tolist(),truemap[:,0].tolist(),truemap[:,1].tolist());
    #X = eng.intersect(matlab.double([[-1,1,1,-1,-1],[-1,-1,1,1,-1]]),matlab.double([[0,10],[0,10]]) )
    X = np.array(X)

    xx = X[1, :]
    if np.linalg.norm(xx - x0[:2]) <= robrng:
        return xx + 0.5 * np.random.randn(2)
    else:
        return None


X = []
for th in np.linspace(0, 2 * np.pi, 50):
    x = x0.copy()
    x[2] = th
    X.append(recordmeas(truemap, x, robrng))

X = np.array(X)
# ax.plot([x0[0],X[0]],[x0[1],X[1]],'k--')
ax.plot(X[:, 0], X[:, 1], 'k*')
plt.show()

d = np.sqrt(np.sum(pow(X[1:, :] - X[0:-1, :], 2), axis=1))
dd = np.hstack((0, d))
T = np.cumsum(dd)
# T=np.arange(len(X[:,0]))
t = np.linspace(0, T[-1], 100)


Xc = X.copy()
vlambda = 2
weights = np.ones(X.shape[0])
Xc[:, 0] = OptTrendFit_cvx(X[:, 0], vlambda, weights)
Xc[:, 1] = OptTrendFit_cvx(X[:, 1], vlambda, weights)


fig2 = plt.figure()
ax2x = fig2.add_subplot(211)
ax2y = fig2.add_subplot(212)
ax2x.plot(T, X[:, 0], 'bo-')
ax2y.plot(T, X[:, 1], 'bo-')
ax2x.plot(T, Xc[:, 0], 'rs-')
ax2y.plot(T, Xc[:, 1], 'rs-')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([-10, 100])
ax.set_ylim([-10, 100])

ax.plot(truemap[:, 0], truemap[:, 1], 'bo-')
plotrobot(x0, ax)
ax.plot(X[:, 0], X[:, 1], 'k*')
ax.plot(Xc[:, 0], Xc[:, 1], 'ro-')
plt.show()
