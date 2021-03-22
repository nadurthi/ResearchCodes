# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import quaternion
import datetime
from scipy.interpolate import UnivariateSpline
from fastdist import fastdist
from scipy.optimize import minimize, rosen, rosen_der,least_squares
from scipy import interpolate
import numpy.linalg as nplinalg
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D


#%%

def runningLowPass(x,fc=np.ones(5)/5):
    """
    output: sum([x1,x2,x3,x4,x5]*fc)

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    fc : TYPE, optional
        DESCRIPTION. The default is np.ones(5)/5.
    shift : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    """
    pass


def quat2omega_quatspline(tvec,qvec,spline_degree=3):
    qqvec=quaternion.from_float_array(qvec)
    # q=quaternion.from_float_array(qvec)
    # ff=quaternion.as_float_array(q)
    # qdot=quaternion.calculus.spline_derivative(qvec,tvec)
    qvec_sp=quaternion.calculus.spline_evaluation(qvec,tvec, axis=0, derivative_order=0,spline_degree=spline_degree)
    qdot=quaternion.calculus.spline_evaluation(qvec_sp,tvec, axis=0, derivative_order=1,spline_degree=spline_degree)
    qqdot=quaternion.from_float_array(qdot)
    w=np.zeros((len(qvec),4))
    for i in range(len(qvec)):
        pp=quaternion.as_float_array(qqdot[i]*qqvec[i].inverse())
        w[i]= 2*pp
    
    return tvec,qvec_sp,w,qdot
    
def quat2omega_scipyspline(tvec,qvec,k=2,s=0.001):
    spl0 = UnivariateSpline(tvec, qvec[:,0], k=k,s=s)
    spl1 = UnivariateSpline(tvec, qvec[:,1], k=k,s=s)
    spl2 = UnivariateSpline(tvec, qvec[:,2], k=k,s=s)
    spl3 = UnivariateSpline(tvec, qvec[:,3], k=k,s=s)
    
    spld0=spl0.derivative()
    spld1=spl1.derivative()
    spld2=spl2.derivative()
    spld3=spl3.derivative()
    
    qdot = np.zeros_like(qvec)
    qvec_sp = np.zeros_like(qvec)
    
    qvec_sp[:,0] = spl0(tvec)
    qvec_sp[:,1] = spl1(tvec)
    qvec_sp[:,2] = spl2(tvec)
    qvec_sp[:,3] = spl3(tvec)
    
    qdot[:,0] = spld0(tvec)
    qdot[:,1] = spld1(tvec)
    qdot[:,2] = spld2(tvec)
    qdot[:,3] = spld3(tvec)
    
    qqdot=quaternion.from_float_array(qdot)
    qqvec=quaternion.from_float_array(qvec_sp)
    
    w=np.zeros((len(qvec),4))
    for i in range(len(qvec)):
        pp=quaternion.as_float_array(qqdot[i]*qqvec[i].inverse())
        w[i]= 2*pp
    
    spl0 = UnivariateSpline(tvec, w[:,0], k=k,s=s)
    spl1 = UnivariateSpline(tvec, w[:,1], k=k,s=s)
    spl2 = UnivariateSpline(tvec, w[:,2], k=k,s=s)
    spl3 = UnivariateSpline(tvec, w[:,3], k=k,s=s)
    
    spld0=spl0.derivative()
    spld1=spl1.derivative()
    spld2=spl2.derivative()
    spld3=spl3.derivative()
    
    alpha = np.zeros((len(qvec),4))
    alpha[:,0] = spld0(tvec)
    alpha[:,1] = spld1(tvec)
    alpha[:,2] = spld2(tvec)
    alpha[:,3] = spld3(tvec)
    
    return tvec,qvec_sp,w,qdot,alpha
    
def quat2omega_poly(tvec,qvec,win=3,poly=3):
    qdot = np.zeros_like(qvec)
    qvec_sp = np.zeros_like(qvec)
    
    for i in range(4):
        Xeval=derivative_poly(tvec,qvec[:,i],orders=1,win=win,poly=poly)
        qvec_sp[:,i]=Xeval[0]
        qdot[:,i]=Xeval[1]

    
    qqdot=quaternion.from_float_array(qdot)
    qqvec=quaternion.from_float_array(qvec_sp)
    
    w=np.zeros((len(qvec),4))
    for i in range(len(qvec)):
        pp=quaternion.as_float_array(qqdot[i]*qqvec[i].inverse())
        w[i]= 2*pp
    
    alpha=np.zeros((len(qvec),4))
    for i in range(4):
        Xeval=derivative_poly(tvec,w[:,i],orders=1,win=win,poly=poly)
        w[:,i]=Xeval[0]
        alpha[:,i]=Xeval[1]
        
    return tvec,qvec_sp,w,qdot,alpha


def derivative_spline(tvec,xvec,teval=None,orders=2,k=2,s=0.001):
        
    spl = UnivariateSpline(tvec, xvec, k=k,s=s)
    # xs = spl(tvec)
    # X=[xs]
    Xeval=[]
    if teval is not None:
        Xeval.append(spl(teval)) 
        
    if orders>=1:
        splv=spl.derivative()
        # xds = splv(tvec)
        # X.append(xds)
        if teval is not None:
            Xeval.append(splv(teval)) 
        
    if orders>=2:
        spla = splv.derivative()
        # xdds = spla(tvec)
        # X.append(xdds)
        if teval is not None:
            Xeval.append(spla(teval)) 
            
    return Xeval


def derivative_poly(tvec,xvec,orders=2,win=3,poly=3):
    
    Xeval=[]
    for i in range(orders+1):
        Xeval.append(np.zeros(len(tvec)))
    
    for i in range(len(tvec)):
        
        n0 = max([i-win,0])
        n1 = min([i+win,len(tvec)-1])
        # print(i,n0,n1)
        tt = tvec[n0:n1]
        xx = xvec[n0:n1]
        t0 = tt[0]
        tt = tt-t0
        pord = min([poly,len(tt)])
        A=np.zeros((len(tt),pord+1))
        A[:,0]=1
        for j in range(1,pord+1):
           A[:,j] = np.power(tt,j)
       
        a = nplinalg.pinv(A).dot(xx)
        pp=np.arange(pord+1)
        # ind = np.logical_and(teval>=tvec[n0],teval<=tvec[n1-1])
        # te = teval[ind]-t0
        te = tvec[i]-t0
        Xeval[0][i]=np.sum(a*np.power(te.reshape(-1,1),pp),axis=1)
        
        
        if orders>=1:
            ad = a*pp
            ppd = pp-1
            ppd[ppd<0]=0
            Xeval[1][i]=np.sum(ad*np.power(te.reshape(-1,1),ppd),axis=1)
            
        if orders>=2:
            add = ad*ppd
            ppdd = ppd-1
            ppdd[ppdd<0]=0
            Xeval[2][i]=np.sum(add*np.power(te.reshape(-1,1),ppdd),axis=1)
    
    
    return Xeval
    
    
    
    
