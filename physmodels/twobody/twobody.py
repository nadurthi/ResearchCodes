# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 23:01:07 2019

@author: Nagnanamus
"""

from numpy import linalg as LA
import numpy as np
import scipy.linalg as sclg
import sys
from scipy.integrate import solve_ivp
from physmodels import motionmodels as phymm


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


class TBP6DODE(phymm.MotionModel):
    motionModelName = 'TBP6D_ODE'
    def __init__(self, mu=1,**kwargs):
        self.mu = mu
        self.fn = 6
        super().__init__(**kwargs)
        
    def contModel(self,t,x,mu):
        r = LA.norm(x[0:2])
        dx = x.copy()
        dx[0] = x[2]
        dx[1] = x[3]
        dx[2] = -(mu / r**3) * x[0]
        dx[3] = -(mu / r**3) * x[1]
    
        return dx
        

    def integrate(self,t_eval,x0):
        sol = solve_ivp(self.contModel, [t_eval[0], t_eval[-1]], x0,t_eval = t_eval,method='RK45',args=(self.mu), rtol=1e-12, atol=1e-12)        
        return sol.y.T
    
    def integrate_batch(self,t_eval,X0):
        S=[]
        for x0 in X0:
            sol=solve_ivp(self.contModel, [t_eval[0], t_eval[-1]], x0,t_eval = t_eval,method='RK45',args=(self.mu), rtol=1e-12, atol=1e-12)        
            S.append(sol.y.T)
        return np.stack(S,axis=0)
    
    
    def propforward(self, t, dt, xk, uk=0, **params):
        sol = solve_ivp(self.contModel, [t, t+dt], xk,method='RK45',args=(self.mu), rtol=1e-12, atol=1e-12)        
        return (t + dt, sol.y.T[-1])

    def processNoise(self,t,dt,xk,uk=0,**params):
        
        Q = dt**2*np.identity(self.fn)

        super().processNoise(**params)
        return Q

    def F(self, t, dt, xk,uk=0, **params):
        super().F(t, dt, xk, uk=None, **params)
        F=None

        return F
    
    
class TBP6DFnG(phymm.MotionModel):
    motionModelName = 'TBP6DFnG'
    def __init__(self, mu=1,**kwargs):
        self.mu = mu
        self.fn = 6
        self.tol = 1e-6
        super().__init__(**kwargs)
        self.states = ['x','y','z','vx','vy','vz']
        
    def contModel(self,t,x,mu):
           
        return None
        

    def integrate(self,t_eval,x0):
        # sol = solve_ivp(self.contModel, [t_eval[0], t_eval[-1]], x0,t_eval = t_eval,method='RK45',args=(self.mu), rtol=1e-12, atol=1e-12)        
        X=np.zeros((len(t_eval),len(x0)))
        X[0,:]=x0
        for i in range(1,len(t_eval)):
            X[i,:] = FnG(t_eval[i-1], t_eval[i], X[i-1,:], self.mu, self.tol)
        return X
    
    def integrate_batch(self,t_eval,X0):
        S=[]
        for x0 in X0:
            X=np.zeros((len(t_eval),len(x0)))
            X[0,:]=x0
            for i in range(1,len(t_eval)):
                X[i,:] = FnG(t_eval[i-1], t_eval[i], X[i-1,:], self.mu, self.tol)
            S.append(X)
        return np.stack(S,axis=0)
    
    
    def propforward(self, t, dt, xk, uk=0, **params):
        X = FnG(t, t+dt, xk, self.mu, self.tol)
        return (t + dt, X)

    def processNoise(self,t,dt,xk,uk=0,**params):
        
        Q = sclg.block_diag([0.00001^2,0.00001^2,0.00001^2,0.0000001^2,0.0000001^2,0.0000001^2])

        super().processNoise(**params)
        return Q

    def F(self, t, dt, xk,uk=0, **params):
        super().F(t, dt, xk, uk=None, **params)
        F=None

        return F    
