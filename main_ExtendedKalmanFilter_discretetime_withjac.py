# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 08:19:18 2021

@author: Nagnanamus
"""
import numpy as np
from numpy.linalg import multi_dot
import numpy.linalg as nplg
import scipy.linalg as sclg
import matplotlib.pyplot as plt
import scipy.optimize as scopt
import pdb
import numba
from numba import vectorize, float64,guvectorize,int64,double,int32,float32
from numba import njit, prange,jit
plt.close("all")



#%% Kalman Filter

# def propagate(tk, dt,xfk,Pfk,uk,Ak,Bk,Qk):
#     xfk1 = np.matmul(Ak,xfk)+np.matmul(Bk,uk)
#     Pfk1 = multi_dot([Ak, Pfk, Ak.T]) + Qk

#     return xfk1, Pfk1


def KalmanDiscreteUpdate(xfk, Pfk, zk, mz, Pz, Pxz):
    K = np.matmul(Pxz, nplg.inv(Pz))
    xu = xfk + np.matmul(K, zk - mz)
    Pu = Pfk - multi_dot([K, Pz, K.T])

    return (xu, Pu,K)

def measUpdate(tk, dt,xfk,Pfk,Hk,Rk,zk):
    """
    if zk is None, just do pseudo update
    return (xu, Pu, mz,R, Pxz,Pz, K, likez )
    """

    mz = np.matmul(Hk,xfk)
    Pxz = np.matmul(Pfk, Hk.T)
    Pz = multi_dot([Hk, Pfk, Hk.T]) + Rk

    xu, Pu, K = KalmanDiscreteUpdate(xfk, Pfk, zk, mz, Pz, Pxz)
    return xu, Pu

def clipvalue(x,maxv,minv):
    return np.min([np.max([x,minv]),maxv])
    

@jit(float64[:](float64,float64[:]),nopython=True, nogil=True,cache=True) 
def CTturnmodel(dt,xk):
    T = dt;
    omg = xk[-1] + 1e-10
    sn=np.sin(omg * T)
    cs=np.cos(omg * T)
    A=np.array([[1, 0, sn / omg, -(1 - cs) / omg, 0],
 				 [0, 1, (1 - cs) / omg, sn / omg, 0],
 				 [0, 0, cs, -sn, 0],
 				 [0, 0, sn, cs, 0],
 				 [0, 0, 0, 0, 1.0]],dtype=np.float64)
    
    xk1 = np.dot(A,xk);

    return xk1

@jit(float64[:,:](float64,float64[:]),nopython=True, nogil=True,cache=True) 
def CTturnmodel_jac(dt,xk):
    T = dt;
    omg = xk[-1] + 1e-10

    if np.abs(omg) >1e-3:

        f=[ np.cos(omg*T)*T*xk[2]/omg - np.sin(omg*T)*xk[2]/omg**2 - np.sin(omg*T)*T*xk[3]/omg - (-1+np.cos(omg*T))*xk[3]/omg**2,
        np.sin(omg*T)*T*xk[2]/omg - (1-np.cos(omg*T))*xk[2]/omg**2 + np.cos(omg*T)*T*xk[3]/omg - np.sin(omg*T)*xk[3]/omg**2,
        -np.sin(omg*T)*T*xk[2] - np.cos(omg*T)*T*xk[3],
        np.cos(omg*T)*T*xk[2] - np.sin(omg*T)*T*xk[3] ]

        A = np.array([[1, 0, np.sin(omg * T) / omg, -(1 - np.cos(omg * T)) / omg, f[0] ],
				 [0, 1, (1 - np.cos(omg * T)) / omg, np.sin(omg * T) / omg, f[1]],
				 [0, 0, np.cos(omg * T), -np.sin(omg * T), f[2]],
				 [0, 0, np.sin(omg * T), np.cos(omg * T), f[3]],
				 [0, 0, 0, 0, 1.0]],dtype=np.float64)

    else:
        A= [ [1.0,0,T,0,-0.5*T**2*xk[3]],
             [0,1.0,0,T,0.5*T**2*xk[2]],
             [0,0,1.0,0,-T*xk[3]],
             [0,0,0,1.0,T*xk[2]],
             [0,0,0,0,1.0]
             ]
        A = np.array(A,dtype=np.float64)

    return A

@jit(float64(float64[::1],float64[:,::1],int32,int32,float64[:],float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64),nopython=True, nogil=True,cache=False) 
def batchEKFcost(XX,Y,statedim,n,x0h,P0,R,Q,H,dt):
    # X= [x0, x1,x2,...xn] len = n+1 timesteps
    # Y= [    y1,y2,...yn] len = n timesteps
    # f is the nonlinear process model
    # X = X.reshape(n+1,statedim)
    X=np.zeros((n+1,statedim))
    for i in range(0,n+1):
        X[i] = XX[i*statedim:(i+1)*statedim]
    J=0.0
    # pdb.set_trace()
    P0inv = nplg.inv(P0)
    Rinv = nplg.inv(R)
    Qinv = nplg.inv(Q)
    
    z = X[0]-x0h 
    a = np.dot(P0inv,z)
    a = np.dot(z,a)
        
    J = a
    
    # adding measurement costs
    for i in range(n):
        z = Y[i]-H.dot(X[i+1])
        a = np.dot(Rinv,z)
        a = np.dot(z,a)
        J +=  a
        
    # adding process model costs
    for i in range(1,n+1):
        z=X[i]-CTturnmodel(dt,X[i-1])
        a = np.dot(Qinv,z)
        a = np.dot(z,a)
        J +=  a
    

    return 0.5*J


@jit(numba.types.Tuple((float64,float64[:]))(float64[::1],float64[:,::1],int32,int32,float64[:],float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64),nopython=True, nogil=True,cache=False) 
def batchEKFcostgrad(XX,Y,statedim,n,x0h,P0,R,Q,H,dt):
    # X= [x0, x1,x2,...xn] len = n+1 timesteps
    # Y= [    y1,y2,...yn] len = n timesteps
    # f is the nonlinear process model
    X=np.zeros((n+1,statedim))
    for i in range(0,n+1):
        X[i] = XX[i*statedim:(i+1)*statedim]
    J=0.0
    Jjac=np.zeros((n+1)*statedim,dtype=np.float64)
    # pdb.set_trace()
    P0inv = nplg.inv(P0)
    Rinv = nplg.inv(R)
    Qinv = nplg.inv(Q)
    
    z = X[0]-x0h 
    a = np.dot(P0inv,z)
    a = np.dot(z,a)               
    J = a
    
    dx=X[1]-CTturnmodel(dt,X[0])
    dF=CTturnmodel_jac(dt,X[0])
    Jjac[0:statedim] = np.dot(P0inv,z)+np.dot(np.dot(-dF.T,Qinv),dx)
                              
    # adding measurement costs
    for i in range(n):
        z = Y[i]-H.dot(X[i+1])
        a = np.dot(Rinv,z)
        a = np.dot(z,a)
        J +=  a
    

    # adding process model costs
    for i in range(1,n):
        dx=X[i]-CTturnmodel(dt,X[i-1])
        a = np.dot(Qinv,dx)
        a = np.dot(dx,a)
        J +=  a
        
        dy= Y[i-1]-H.dot(X[i])
        dxf=X[i+1]-CTturnmodel(dt,X[i])
        dF=CTturnmodel_jac(dt,X[i])
        Jjac[i*statedim:(i+1)*statedim] =  np.dot(np.dot(-H.T,Rinv),dy)+np.dot(Qinv,dx) +np.dot(np.dot(-dF.T,Qinv),dxf)
    
    # for the very last state xn
    dx=X[n]-CTturnmodel(dt,X[n-1])
    a = np.dot(Qinv,dx)
    a = np.dot(dx,a)
    J +=  a
    
    dy= Y[n-1]-H.dot(X[n])
    Jjac[n*statedim:(n+1)*statedim] =  np.dot(np.dot(-H.T,Rinv),dy)+np.dot(Qinv,dx)

    
    return 0.5*J,Jjac


def batchEKFoptimize(Y,statedim,n,x0h,P0,R,Q,H,dt,X0=None):
    tsteps = len(Y)
    if X0 is None:
        X0=np.zeros((n+1,statedim),dtype=np.float64)
        X0[0]=x0h
        
        for i in range(1,tsteps):
            X0[i] = CTturnmodel(dt,X0[i-1])
            
            
    X0 = X0.reshape((n+1)*statedim)
    # pdb.set_trace()
    # res=scopt.minimize(batchEKFcost,X0,args=(Y,statedim,n,x0h,P0,R,Q,H,dt),method='BFGS')    
    res=scopt.minimize(batchEKFcostgrad,X0,args=(Y,statedim,n,x0h,P0,R,Q,H,dt),jac=True,method='BFGS') 
    
    X=res['x']
    X = X.reshape(n+1,statedim)
    
    return X

# X0=np.array([0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],dtype=np.float64)
# X0=np.random.randn(30).astype(np.float64)
# Y=np.array([[1.59349604, 1.61459039],
#         [2.14592629, 2.27518671],
#         [2.65115329, 2.91662273],
#         [3.05028921, 3.64859484],
#         [3.51849481, 4.29638534]],dtype=np.float64)
# P0=np.array([[2.5e-01, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
#         [0.0e+00, 2.5e-01, 0.0e+00, 0.0e+00, 0.0e+00],
#         [0.0e+00, 0.0e+00, 1.0e-02, 0.0e+00, 0.0e+00],
#         [0.0e+00, 0.0e+00, 0.0e+00, 1.0e-02, 0.0e+00],
#         [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 1.0e-04]],dtype=np.float64)
# statedim=np.int32(5)
# n=np.int32(5)
# x0h=np.array([0., 0., 1., 0., 0.],dtype=np.float64)
# R=np.array([[0.0025, 0.    ],
#         [0.    , 0.0025]],dtype=np.float64)
# Q=np.array([[0.00018, 0.     , 0.0009 , 0.     , 0.     ],
#         [0.     , 0.00018, 0.     , 0.0009 , 0.     ],
#         [0.0009 , 0.     , 0.006  , 0.     , 0.     ],
#         [0.     , 0.0009 , 0.     , 0.006  , 0.     ],
#         [0.     , 0.     , 0.     , 0.     , 0.006  ]],dtype=np.float64)
# H=np.array([[1., 0., 0., 0., 0.],
#         [0., 1., 0., 0., 0.]],dtype=np.float64)
# dt=0.3
# print(batchEKFcost(X0,Y,statedim,n,x0h,P0,R,Q,H,dt))
# # print(batchEKFcost_prev(X0,Y,statedim,n,x0h,P0,R,Q,H,dt))
# batchEKFcostgrad(X0,Y,statedim,n,x0h,P0,R,Q,H,dt)
#%% PLotting functions

def getEllipsePoints2D(xyc,a,b,N=100):
    th = np.linspace(0,2*np.pi,N)
    X = np.zeros((N,2))
    for i in range(len(th)):    
        X[i,:]=xyc+[a*np.cos(th[i]),b*np.sin(th[i])];
    
    return X



def getCirclePoints2D(xyc,r,N=100):
    X = getEllipsePoints2D(xyc,r,r,N=N)
    return X


def getCovEllipsePoints2D(m,P,nsig=1,N=100):
    mz = np.zeros(m.shape)
    A = sclg.sqrtm(P) 
    X = getCirclePoints2D(mz,nsig,N=N)
    
    for i in range(X.shape[0]):    
        X[i,:]=m+np.matmul(A,X[i,:])
    
    return X



#%% Model Parameters

dt=T=0.3 # time step
Bk=np.array([[0,0,1,1,0],[0,0,0,0,1]],dtype=np.float64).T
dim=fn=5

L1=0.02
L2=0.02

Q_CT = L1 * np.array([[T**3 / 3, 0, T**2 / 2, 0, 0],
		 [0, T**3 / 3, 0, T**2 / 2, 0], [
		 T**2 / 2, 0, T, 0, 0], [
			0, T**2 / 2, 0, T, 0], [
		 0, 0, 0, 0, T * L2 / L1]],dtype=np.float64)


# linear model for measurements: measure x and y
Hk=np.array([[1,0,0,0,0],[0,1,0,0,0]],dtype=np.float64)
Rk=np.array([[0.05**2,0],[0,0.05**2]],dtype=np.float64)
hn=2     


#%% generate a ground truth i.e. measurements
N=100
xt=np.array([1,1,2,2,0.2],dtype=np.float64)
Xt=np.zeros((N,dim),dtype=np.float64)
Zk=np.zeros((N,hn),dtype=np.float64)
Xt[0]=xt

for i in range(1,N):
    Xt[i]=CTturnmodel(dt,Xt[i-1])
    


    s=np.random.rand(1)[0]
    if s>0.9:
        Omegadt = 2*np.random.rand(1)[0]-1
        magk = 2*np.random.rand(1)[0]
        
        Omegadt = clipvalue(Omegadt,0.1,-0.1)
        mag = clipvalue(magk,1.3,0.7)
        
        Xt[i,2:4] = mag*Xt[i,2:4]
        Xt[i,4] = Xt[i,4]+ Omegadt
        
    # geenrate some random measurements
    Zk[i]=np.matmul(Hk,Xt[i])+np.matmul(sclg.sqrtm(Rk), np.random.randn(hn))
        
        

plt.figure()
plt.plot(Xt[:,0],Xt[:,1],'k*-')  
plt.plot(Zk[1:,0],Zk[1:,1],'r*')
        
        
#%%    Now run Kalman filter only with measurements and no knowledge of the ground truth
plt.close("all")
# make a guess of the initial start point
xfk= x0 = np.array([0,0,1,0,0],dtype=np.float64)
Pfk= P0 = np.array([[5**2,0,0,0,0],
               [0,5**2,0,0,0],
               [0,0,2**2,0,0],
               [0,0,0,2**2,0],
               [0,0,0,0,0.5**2]],dtype=np.float64) 


# save before measurement update
Xfk_prior=np.zeros((N,dim),dtype=np.float64)
Covfk_prior=np.zeros((N,dim,dim),dtype=np.float64)

Xfk_prior[0]=xfk
Covfk_prior[0]=Pfk

# save after measurement update
Xfk=np.zeros((N,dim),dtype=np.float64)
Covfk=np.zeros((N,dim,dim),dtype=np.float64)

Xfkbatch=np.zeros((N,dim),dtype=np.float64)

Xfk[0]=xfk
Covfk[0]=Pfk
Xfkbatch[0]=xfk

# assuming we have no idea what control was used on the robot
uk=np.array([0],dtype=np.float64)  

# fig=plt.figure("Estimated trajectory")
# ax = fig.add_subplot(111)

fig2=plt.figure("Estimated trajectory with covariance")
ax2 = fig2.add_subplot(111)

# now run as if we are in real-time 
for i in range(1,N):
    
    # predict the mean and covariance
    xfk1 = CTturnmodel(dt,xfk)
    Ak =  CTturnmodel_jac(dt,xfk)
    Pfk1 = multi_dot([Ak, Pfk, Ak.T]) + Q_CT
    
    Xfk_prior[i]=xfk1
    Covfk_prior[i]=Pfk1
    
    # do the measurmenet update with the measurement at this time
    zk = Zk[i]
    xu, Pu = measUpdate(dt*i, dt,xfk1,Pfk1,Hk,Rk,zk)

    Xfk[i]=xu
    Covfk[i]=Pu
    
       
    # now copy the updated values for the next iteration
    
    xfk = xu
    Pfk = Pu
    
    
    # ax.cla()
    # ax.plot(Xt[:i+1,0],Xt[:i+1,1],'k*-',label='Truth')  
    # ax.plot(Zk[1:i+1,0],Zk[1:i+1,1],'r*',label='measurements')
    # ax.plot(Xfk[:i+1,0],Xfk[:i+1,1],'b.-',label="estimate")  
    # ax.set_xlim(np.min(Xt[:,0])-10,np.max(Xt[:,0])+10)
    # ax.set_ylim(np.min(Xt[:,1])-10,np.max(Xt[:,1])+10)
    # ax.legend()
    
    # ax.set_xlim(np.min(Xt[:,0])-10,np.max(Xt[:,0])+10)
    # ax.set_ylim(np.min(Xt[:,1])-10,np.max(Xt[:,1])+10)
    # ax.legend()
    # plt.show()
    
    ax2.cla()
    ax2.plot(Xt[:i+1,0],Xt[:i+1,1],'k*-',label='Truth')  
    ax2.plot(Zk[1:i+1,0],Zk[1:i+1,1],'r*',label='measurements')
    ax2.plot(Xfk[:i+1,0],Xfk[:i+1,1],'b.-',label="estimate")
    
    Xe= getCovEllipsePoints2D(Xfk_prior[i][0:2],Covfk_prior[i][0:2,0:2],nsig=1,N=100)
    ax2.plot(Xe[:,0],Xe[:,1],'g--')
    
    Xe= getCovEllipsePoints2D(Xfk[i][0:2],Covfk[i][0:2,0:2],nsig=1,N=100)
    ax2.plot(Xe[:,0],Xe[:,1],'g')
    
    ax2.set_xlim(np.min(Xt[:,0])-10,np.max(Xt[:,0])+10)
    ax2.set_ylim(np.min(Xt[:,1])-10,np.max(Xt[:,1])+10)
    ax2.legend()
    
    n=30# the number of timesteps you want to got back
    if i >= n:
        # last n measurements
        x0h = Xfkbatch[i-(n-1)-1]
        P0h = np.array([[0.5**2,0,0,0,0],
               [0,0.5**2,0,0,0],
               [0,0,0.1**2,0,0],
               [0,0,0,0.1**2,0],
               [0,0,0,0,0.01**2]]) 

        X0guess = Xfkbatch[i-(n-1)-1:i+1]
        X0guess[-1] = CTturnmodel(dt,X0guess[-2])
        XX = batchEKFoptimize(Zk[i-(n-1):i+1],fn,n,x0h,P0h,Rk,Q_CT,Hk,dt,X0=X0guess)
        Xfkbatch[i-(n-1)-1:i+1]=XX    
        
        
                  

            
        ax2.plot(Xfkbatch[:i+1,0],Xfkbatch[:i+1,1],'ms-',label='BatchEst') 
        


    plt.show()
    plt.pause(0.2)
    
fig3=plt.figure("Estimated Turn-rate")
ax3 = fig3.add_subplot(111)
ax3.plot(dt*np.arange(0,N),Xt[:,4],label="True ground truth turn rate ")
ax3.plot(dt*np.arange(0,N),Xfk[:,4],label="Estimated turn rate ")
ax3.plot(dt*np.arange(0,N),Xfkbatch[:,4],label="Batch Estimated turn rate ")
ax3.legend()
plt.show()


