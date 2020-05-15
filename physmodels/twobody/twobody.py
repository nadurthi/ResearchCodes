# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 23:01:07 2019

@author: Nagnanamus
"""

from numpy import linalg as LA
import numpy as np
import sys
from scipy.integrate import solve_ivp


def kepler(M, e, tol):

    #% Compute Eccentric Anomayl
    if (M > -np.pi and M < 0 or M > np.pi):
        E1 = M - e
    elif (M < -np.pi or M > 0 and M < np.pi):
        E1 = M + e
    elif (M == 0):
        E1 = M + e

    E = E1 + (M - E1 + e * np.sin(E1)) / (1 - e * np.cos(E1))

    while (abs(E - E1) > tol):
        E1 = E
        E = E1 + (M - E1 + e * np.sin(E1)) / (1 - e * np.cos(E1))

    E = (E / (2 * np.pi) - np.floor(E / (2 * np.pi))) * 2 * np.pi

    # Compute True Anomaly
    f = 2 * np.arctan2(np.sqrt(1 + e) * np.tan(E / 2), np.sqrt(1 - e))

    return [E, f]


def FnG(t0, tf, x0, mu, tol):
    r0 = x0[0:3]
    v0 = x0[3:6]

    R0 = LA.norm(r0)  # % Magnitude of current position
    V0 = LA.norm(v0)  # % Magnitude of current velocity
    sigma0 = np.dot(r0, v0) / np.sqrt(mu)  # % Defined
    A = 2 / R0 - V0**2 / mu  # % Reciprocal of 1/a
    a = 1 / A  # % Semi-major axis
    M = np.sqrt(mu / a**3) * (tf - t0)  # % Mean anomaly (rad)

    h = np.cross(r0, v0)
    evec = np.cross(v0, h) / mu - r0 / LA.norm(r0)
    e = LA.norm(evec)

    if e > 1:
        sys.exit('this FNG solution is not valid for e>1 ')

#    % Run Newton-Raphson Method
    tol = 1e-12
    itr = 0
    MaxIt = 100
    Ehat = M

    dEhat = 1
    while abs(dEhat) > tol:
        err = M - (Ehat - (1 - R0 / a) * np.sin(Ehat) +
                   sigma0 / np.sqrt(a) * (1 - np.cos(Ehat)))
        derr = -1 + (1 - R0 / a)*np.cos(Ehat)-(sigma0/np.sqrt(a))*np.sin(Ehat)
        dEhat = np.max([-1, np.min([1, err / derr])])

        Ehat = Ehat - dEhat
        itr = itr + 1
        if itr > MaxIt:
            print('hitting max iter for FnG, Switch to alternative method')
            break

    R = a + (R0 - a) * np.cos(Ehat) + np.sqrt(a) * sigma0 * np.sin(Ehat)
    F = 1 - a / R0 * (1 - np.cos(Ehat))
    G = (tf - t0) + np.sqrt(a**3 / mu) * (np.sin(Ehat) - Ehat)
    Fdot = - np.sqrt(mu * a) / (R * R0) * np.sin(Ehat)
    Gdot = 1 - a / R * (1 - np.cos(Ehat))
    r = F * r0 + G * v0
    v = Fdot * r0 + Gdot * v0

    return np.hstack((r, v))


def twobody2D_ode(t, x, mu):
    r = LA.norm(x[0:2])
    dx = x.copy()
    dx[0] = x[2]
    dx[1] = x[3]
    dx[2] = -(mu / r**3) * x[0]
    dx[3] = -(mu / r**3) * x[1]

    return dx


class PropTBP_cart2D:

    def __init__(self, mu, tol=1e-12):
        self.mu = mu
        self.tol = tol

    def propcart_fwdstep(self, t, x, dt):
        t_span = (t, t + dt)
        sol = solve_ivp(lambda ty, y: twobody2D_ode(ty, y, self.mu), t_span, x,
                        method='RK45', rtol=1e-10, atol=self.tol)
        return sol.y[:, -1]

    def propcart(self, x0, t0, tf):
        t_span = (t0, tf)
        sol = solve_ivp(lambda ty, y: twobody2D_ode(ty, y, self.mu), t_span,
                        x0, method='RK45', rtol=1e-10, atol=self.tol)
        return sol.y[:, -1]

    def propcart_fwdvec(self, x, tvec):
        #        x is at tvec[0]
        t_span = (tvec[0], tvec[-1])
        sol = solve_ivp(lambda ty, y: twobody2D_ode(ty, y, self.mu), t_span,
                        x, method='RK45', t_eval=tvec, rtol=1e-10,
                        atol=self.tol)
        return sol.y.T

    def propcart_fwdbatch(self, X0, t0, tf):
        # x is at tvec[0]
        N, dim = X0.shape
        X = np.zeros((N, dim))
        for i in range(N):
            y = np.array([X0[i, 0], X0[i, 1], 0, X0[i, 2], X0[i, 3], 0])
            X[i, :] = self.propcart(self, y, t0, tf)
        return X

    def propFnG_fwdstep(self, t, x, dt):
        y = np.array([x[0], x[1], 0, x[2], x[3], 0])
        yf = FnG(t, t + dt, y, self.mu, self.tol)
        xf = yf[[0, 1, 3, 4]]
        return xf

    def propFnG(self, x, t0, tf):
        y = np.array([x[0], x[1], 0, x[2], x[3], 0])
        yf = FnG(t0, tf, y, self.mu, self.tol)
        xf = yf[[0, 1, 3, 4]]
        return xf

    def propFnG_fwdvec(self, x, tvec):
        # x is at tvec[0]
        N = len(tvec)
        dim = len(x)
        X = np.zeros((N, dim))
        X[0, :] = x
        for i in range(N - 1):
            y = np.array([X[i, 0], X[i, 1], 0, X[i, 2], X[i, 3], 0])
            yf = FnG(tvec[i], tvec[i + 1], y, self.mu, self.tol)
            X[i + 1, :] = yf[[0, 1, 3, 4]]
        return X

    def propFnG_fwdbatch(self, X0, t0, tf):
        # x is at tvec[0]
        N, dim = X0.shape
        X = np.zeros((N, dim))
        for i in range(N):
            y = np.array([X0[i, 0], X0[i, 1], 0, X0[i, 2], X0[i, 3], 0])
            yf = FnG(t0, tf, y, self.mu, self.tol)
            X[i, :] = yf[[0, 1, 3, 4]]
        return X
