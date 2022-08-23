# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd

from ssatoolkit import coords
from numpy import linalg as nplinalg
import numpy as np
import scipy.linalg as sclg
import sys
from scipy.integrate import solve_ivp
import numba as nb
from scipy.integrate import odeint



def propagate_kep(r, v, Delta_t, mu,step=1):
    prop_r = np.empty((0, 3))
    prop_v = np.empty((0, 3))
    # mu = 398600  # km3/s2
    Ihat = np.array([1, 0, 0])
    Jhat = np.array([0, 1, 0])
    Khat = np.array([0, 0, 1])

    h = np.cross(r, v)  # Angular momentum vector
    e = np.cross(v, h) / mu - r / np.linalg.norm(r)  # Eccentricity vector
    n = np.cross(Khat, h)  # Node vector

    # Let's now find i, Omega, omega and the vu. They are the orbital elements of our initial orbit
    i = np.arccos(h.dot(Khat) / np.linalg.norm(h))  # Inclination
    Om = np.arccos(n.dot(Ihat) / np.linalg.norm(n))  # Ascending node
    om = np.arccos(n.dot(e) / (np.linalg.norm(n) * np.linalg.norm(e)))  # periapsis position
    f = np.arccos(e.dot(r) / (np.linalg.norm(e) * np.linalg.norm(r)))  # True anomaly
    a = np.linalg.norm(r) / (2 - ((np.linalg.norm(r) * np.linalg.norm(v) ** 2) / mu))
    Orbit1 = np.array([a, np.linalg.norm(e), i, Om, om, f])

    # Now that we have all the orbital elements, we must find the future true anomaly of the next step, and all the following steps:
    E = np.arccos((Orbit1[1] + np.cos(Orbit1[5])) / (1 + Orbit1[1] * np.cos(Orbit1[5])))  # Mean eccentricity
    current_Dt = np.sqrt(Orbit1[0] ** 3 / mu) * (
                E - Orbit1[1] * np.sin(E))  # We compute the delta t of the first observation
    i = 0
    while current_Dt + Delta_t > current_Dt + step * i:
        M = math.sqrt(mu / Orbit1[0] ** 3) * (
                    current_Dt + step * i)  # We compute the mean anomaly at the desired future position

        E = Find_Kep_E(Orbit1[1], M)
        Orbit1[5] = np.arccos((np.cos(E) - Orbit1[1]) / (1 - Orbit1[1] * np.cos(E)))

        if (int(E / math.pi) % 2) == 1:
            Orbit1[5] = 2 * math.pi - Orbit1[5]

        # And finally, find the new r and v:
        p = Orbit1[0] * (1 - Orbit1[1] ** 2)

        r[0] = p * np.cos(Orbit1[5]) / (
                    1 + Orbit1[1] * np.cos(Orbit1[5]))  # Here we compute the coordinates in the PQW coordinate system
        r[1] = p * np.sin(Orbit1[5]) / (1 + Orbit1[1] * np.cos(Orbit1[5]))
        r[2] = 0
        v[0] = -np.sqrt(mu / p) * np.sin(Orbit1[5])
        v[1] = np.sqrt(mu / p) * (Orbit1[1] + np.cos(Orbit1[5]))
        v[2] = 0

        prop_r = np.append(prop_r, coords.pqwTOijk(r, -Orbit1[3], -Orbit1[4],
                                            -Orbit1[2]))  # We Convert the PQW coordinates to XYZ coordinates
        prop_r = np.reshape(prop_r, (i + 1, 3))
        prop_v = np.append(prop_v, coords.pqwTOijk(v, -Orbit1[3], -Orbit1[4], -Orbit1[2]))
        prop_v = np.reshape(prop_v, (i + 1, 3))

        i = i + 1
    return prop_r, prop_v

def two_body_eqm(_y, _t, _mu):

    r_mag = np.linalg.norm(_y[:3])
    c0 = _y[3:6]
    c1 = -_mu * ((_y[:3]) / np.power(r_mag, 3))
    return np.concatenate((c0, c1))


# ==============================================================
# simulation harness
def propagate_orb(tvec, orb_elem,mu):
    #tvec[0] has to be the initial time for the orb_elem
    # mu = 398600  # km3/s2
    # time = np.arange(t0, tf, dt)
    y0 = coords.from_OE_to_RV(orb_elem,mu)
    y = odeint(two_body_eqm, y0, tvec, args=(mu,),rtol=1e-12,atol=1e-12)
    return y

# simulation harness
def propagate_rv(t10, r10, v10, tf, mu):
    # mu = 398600  # km3/s2
    time =[t10,tf]
    y0 = np.concatenate((r10, v10))
    yf = odeint(two_body_eqm, y0, time, args=(mu,),rtol=1e-12,atol=1e-12)
    return yf

def propagate_rv_tvec(t0, rv0, tvec, mu):
    # mu = 398600  # km3/s2
    tvec = np.sort(tvec)
    X=np.zeros((len(tvec),6))
    if tvec[0]==t0:
        time =tvec
        X = odeint(two_body_eqm, rv0, time, args=(mu,),rtol=1e-12,atol=1e-12)
    
    if tvec[-1]==t0:
        time =tvec[::-1]
        X = odeint(two_body_eqm, rv0, time, args=(mu,),rtol=1e-12,atol=1e-12)
        X=X[::-1,:]
            
    if tvec[0]>t0:
        time =np.hstack([t0,tvec])
        X = odeint(two_body_eqm, rv0, time, args=(mu,),rtol=1e-12,atol=1e-12)
        X=X[1:,:]
        
        
    if tvec[-1]<t0:
        time =np.hstack([t0,tvec[::-1]])
        X = odeint(two_body_eqm, rv0, time, args=(mu,),rtol=1e-12,atol=1e-12)
        X=X[1:,:]
        X=X[::-1,:]
    
    if t0>tvec[0] and t0<tvec[-1]:
        i0 = np.where(tvec>=t0)[0][0]
        if tvec[i0]==t0:
            X[i0]=rv0
            time =tvec[i0:]
            X1 = odeint(two_body_eqm, rv0, time, args=(mu,),rtol=1e-12,atol=1e-12)
            
            time =tvec[:i0+1][::-1]
            X2 = odeint(two_body_eqm, rv0, time, args=(mu,),rtol=1e-12,atol=1e-12)
            X2=X2[1:,:]
            X2=X2[::-1,:]
        else:
            time =np.hstack([t0,tvec[i0:]])
            X1 = odeint(two_body_eqm, rv0, time, args=(mu,),rtol=1e-12,atol=1e-12)
            X1=X1[1:,:]

            time =np.hstack([t0,tvec[:i0][::-1]])
            X2 = odeint(two_body_eqm, rv0, time, args=(mu,),rtol=1e-12,atol=1e-12)
            X2=X2[1:,:]
            X2=X2[::-1,:]
        
        X[:i0,:]=X2
        X[i0:,:]=X1
        
    
    return tvec,X


def Find_Kep_E(e,M):
    """ This function solve the kepler's equation by finding the right E.

    Parameters
    ----------

    e: float
      eccentricity of the orbit.
    M: float
      Mean mottion of the orbit.

    Returns
    -------

    E: float
      The mean eccentricity that is solution of the equation.

    """
    i = 0
    tol = 1E-12
    B = np.cos(e) - ((math.pi/2) - e)*np.sin(e)
    E = np.array([])
    fE = np.array([])
    dfE = np.array([])
    d2fE = np.array([])
    E1 = np.array([])
    E2 = np.array([])
    E = np.append(E, M + (e*np.sin(M))/(B+M*np.sin(e)))
    while np.abs(E[i] - M - e*np.sin(E[i])) > tol:

        fE = np.append(fE, E[i] - e*np.sin(E[i]) - M)
        dfE = np.append(dfE,  1 - e*np.cos(E[i]))
        d2fE = np.append(d2fE, e*np.sin(E[i]))

        A = 2*np.sqrt((np.abs(4*(dfE[i]**2))))
        E1 = np.append(E1, E[i] - (5*fE[i]/(dfE[i] + A)))
        E2 = np.append(E2, E[i] - (5*fE[i]/(dfE[i] - A)))

        if (abs(E1[i] - E[i])) < (abs(E2[i] - E[i])):
            E = np.append(E, E1[i])
        else:
                E = np.append(E, E2[i])
        i = i + 1

    if i > 100:
        print(" Kepler's Equation is NOT converging ! ")
        return
    return E[i]






#%%   
##########################################3
##########     Redundant      ############3 
##########################################3



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
    if E<0:
        E=E+2*np.pi
    if E>=2*np.pi:
        E=E-2*np.pi
    
    if f<0:
        f=f+2*np.pi
    if f>=2*np.pi:
        f=f-2*np.pi
            
        
    return [E, f]


def FnG(t0, tf, x0, mu, tol = 1e-12):
    r0 = x0[0:3]
    v0 = x0[3:6]

    R0 = nplinalg.norm(r0)  # % Magnitude of current position
    V0 = nplinalg.norm(v0)  # % Magnitude of current velocity
    sigma0 = np.dot(r0, v0) / np.sqrt(mu)  # % Defined
    A = 2 / R0 - V0**2 / mu  # % Reciprocal of 1/a
    a = 1 / A  # % Semi-major axis
    M = np.sqrt(mu / a**3) * (tf - t0)  # % Mean anomaly (rad)

    h = np.cross(r0, v0)
    evec = np.cross(v0, h) / mu - r0 / nplinalg.norm(r0)
    e = nplinalg.norm(evec)

    if e > 1:
        # print("FnG does not work for e>1, so doing integration")
        x=propagate_rv(t0, x0[:3], x0[3:], tf, mu)
        xf=x[-1]
        return xf
    
#    % Run Newton-Raphson Method

    itr = 0
    MaxIt = 100
    Ehat = M

    dEhat = 1
    flgfail = False
    while abs(dEhat) > tol:
        err = M - (Ehat - (1 - R0 / a) * np.sin(Ehat) +
                   sigma0 / np.sqrt(a) * (1 - np.cos(Ehat)))
        derr = -1 + (1 - R0 / a)*np.cos(Ehat)-(sigma0/np.sqrt(a))*np.sin(Ehat)
        dEhat = np.max([-1, np.min([1, err / derr])])

        Ehat = Ehat - dEhat
        itr = itr + 1
        if itr > MaxIt:
            print('hitting max iter for FnG, Switch to alternative method')
            flgfail =True
            break
    
    if flgfail:
        x=propagate_rv(t0, x0[:3], x0[3:], tf, mu)
        xf=x[-1]
    else:
        R = a + (R0 - a) * np.cos(Ehat) + np.sqrt(a) * sigma0 * np.sin(Ehat)
        F = 1 - a / R0 * (1 - np.cos(Ehat))
        G = (tf - t0) + np.sqrt(a**3 / mu) * (np.sin(Ehat) - Ehat)
        Fdot = - np.sqrt(mu * a) / (R * R0) * np.sin(Ehat)
        Gdot = 1 - a / R * (1 - np.cos(Ehat))
        r = F * r0 + G * v0
        v = Fdot * r0 + Gdot * v0
        xf = np.hstack((r, v))
        
        
    return xf

def propagate_FnG(tvec, x0, mu, tol = 1e-12):
    X=np.zeros((len(tvec),6))
    X[0]=x0
    for i in range(1,len(tvec)):
        X[i]=FnG(tvec[i-1], tvec[i], X[i-1], mu, tol = tol)
    
    return X

def propagate_FnG_mixed(t0,x0,tvec, mu, tol = 1e-12):
    # given t0,x0, this will generate the rv for the tvec, 
    # tvec can also be randomized, does not have to be monotonic
    tvec = np.sort(tvec)
    X=np.zeros((len(tvec),6))
    if tvec[0]==t0:
        X[0]=x0
        for i in range(1,len(tvec)):
            X[i]=FnG(tvec[i-1], tvec[i], X[i-1], mu, tol = tol)
    
    if tvec[-1]==t0:
        X[-1]=x0
        for i in range(len(tvec)-1,0,-1):
            X[i-1]=FnG(tvec[i], tvec[i-1], X[i], mu, tol = tol)
            
    if tvec[0]>t0:
        X[0]=FnG(t0, tvec[0], x0, mu, tol = tol)
        for i in range(1,len(tvec)):
            X[i]=FnG(tvec[i-1], tvec[i], X[i-1], mu, tol = tol)
    
    if tvec[-1]<t0:
        X[-1]=FnG(t0, tvec[-1], x0, mu, tol = tol)
        for i in range(len(tvec)-1,0,-1):
            X[i-1]=FnG(tvec[i], tvec[i-1], X[i], mu, tol = tol)
    
    if t0>tvec[0] and t0<tvec[-1]:
        i0 = np.where(tvec>=t0)[0][0]
        X[i0]=FnG(t0, tvec[i0], x0, mu, tol = tol)
        for i in range(i0+1,len(tvec)):
            X[i]=FnG(tvec[i-1], tvec[i], X[i-1], mu, tol = tol)
        for i in range(i0,0,-1):
            X[i-1]=FnG(tvec[i], tvec[i-1], X[i], mu, tol = tol)
            
    
    return tvec,X
      
def twobody6D_ode(t, x, mu):
    r = nplinalg.norm(x[0:3])
    dx = np.zeros(6)
    dx[0] = x[3]
    dx[1] = x[4]
    dx[2] = x[5]
    dx[3] = -(mu / r**3) * x[0]
    dx[4] = -(mu / r**3) * x[1]
    dx[5] = -(mu / r**3) * x[2]
    return dx

def twobody2D_ode(t, x, mu):
    r = nplinalg.norm(x[0:2])
    dx = x.copy()
    dx[0] = x[2]
    dx[1] = x[3]
    dx[2] = -(mu / r**3) * x[0]
    dx[3] = -(mu / r**3) * x[1]

    return dx


def propagate6D_cart(tk,dt,xk,mu,method='ode'):
    """
    x0 is in cartesian 
    """
    t_eval=np.array([tk,tk+dt])
    if method == 'ode':
        sol = solve_ivp(twobody6D_ode, [tk, tk+dt], xk,t_eval = t_eval,method='RK45',args=(mu,), rtol=1e-12, atol=1e-12)        
        xk1 = sol.y.T[-1,:]
        return xk1
    elif method == 'FnG':
        xk1 = FnG(tk, tk+dt, xk, mu)
        return xk1
    else:
        return None
        

def integrate6D_cart(t_eval,x0,mu,method='ode'):
    """
    x0 is in cartesian 
    """
    if method == 'ode':
        sol = solve_ivp(twobody6D_ode, [t_eval[0], t_eval[-1]], x0,t_eval = t_eval,method='RK45',args=(mu,), rtol=1e-12, atol=1e-12)        
        return sol.y.T
    elif method == 'FnG':
        X=np.zeros((len(t_eval),len(x0)))
        X[0,:]=x0
        for i in range(1,len(t_eval)):
            X[i,:] = FnG(t_eval[i-1], t_eval[i], X[i-1,:], mu)
        return X
    else:
        return None
            

def integrate6D_batch_cart(t_eval,X0,mu,method='ode'):
    """
    x0 is in cartesian 
    """
    S=np.zeros((X0.shape[0],len(t_eval),X0.shape[1]))
    if method == 'ode':
        for i in range(X0.shape[0]):
            sol=solve_ivp(twobody6D_ode, [t_eval[0], t_eval[-1]], X0[i],t_eval = t_eval,method='RK45',args=(mu,), rtol=1e-12, atol=1e-12)        
            S[i] = sol.y.T
        return S
    elif method == 'FnG':
        for i in range(X0.shape[0]):
            S[0,0,:]=X0[i]
            for k in range(1,len(t_eval)):
                S[i,k,:] = FnG(t_eval[k-1], t_eval[k], S[i,k-1,:], mu)
        return S
    else:
        return None
