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


plt.close("all")
        
#%% Kalman Filter

def propagate(tk, dt,xfk,Pfk,uk,Ak,Bk,Qk):
    xfk1 = np.matmul(Ak,xfk)+np.matmul(Bk,uk)
    Pfk1 = multi_dot([Ak, Pfk, Ak.T]) + Qk

    return xfk1, Pfk1


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
    


def CTturnmodel(dt,xk):
    T = dt;
    omg = xk[-1] + 1e-10

    xk1 = np.matmul(np.array([[1, 0, np.sin(omg * T) / omg, -(1 - np.cos(omg * T)) / omg, 0],
				 [0, 1, (1 - np.cos(omg * T)) / omg, np.sin(omg * T) / omg, 0],
				 [0, 0, np.cos(omg * T), -np.sin(omg * T), 0],
				 [0, 0, np.sin(omg * T), np.cos(omg * T), 0],
				 [0, 0, 0, 0, 1]]), xk);
    
    return xk1

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
				 [0, 0, 0, 0, 1]])

    else:
        A= [ [1,0,T,0,-0.5*T**2*xk[3]],
             [0,1,0,T,0.5*T**2*xk[2]],
             [0,0,1,0,-T*xk[3]],
             [0,0,0,1,T*xk[2]],
             [0,0,0,0,1]
             ]
        A = np.array(A)

    return A

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
Ak=np.array([[1, 0, T, 0],  
             [0, 1, 0, T],
             [0, 0, 1, 0],
             [0, 0, 0, 1]])
Bk=np.array([0,0,1,1]).reshape(-1,1)
dim=fn=4

L1=0.2

Qk=L1 * np.array([[T**3 / 3, 0, T**2 / 2, 0], 
                  [0, T**3 / 3, 0, T**2 / 2], 
                  [ T**2 / 2, 0, T, 0], 
                  [0, T**2 / 2, 0, T]])


Hk=np.array([[1,0,0,0],[0,1,0,0]])
Rk=np.array([[0.5**2,0],[0,0.5**2]])
hn=2     


#%% generate a ground truth i.e. measurements
N=100
xt=np.array([3,3,2,2])
Xt=np.zeros((N,dim))
Zk=np.zeros((N,hn))
Xt[0]=xt

for i in range(1,N):
    Xt[i]=np.matmul(Ak,Xt[i-1])


    s=np.random.rand(1)[0]
    if s>0.8:
        thk = np.pi/3*np.random.rand(1)[0]-np.pi/6
        magk = 2*np.random.rand(1)[0]-1
        
        th = np.arctan2(Xt[i,3],Xt[i,2])+thk
        mag = nplg.norm(Xt[i,:2])+magk
        mag = clipvalue(mag,5,2)
        Xt[i,2:] = mag*np.array([np.cos(th),np.sin(th)])  
        
    # geenrate some random measurements
    Zk[i]=np.matmul(Hk,Xt[i])+np.matmul(sclg.sqrtm(Rk), np.random.randn(hn))
        
        

plt.figure()
plt.plot(Xt[:,0],Xt[:,1],'k*-')  
plt.plot(Zk[1:,0],Zk[1:,1],'r*')
        
        
#%%    Now run Kalman filter only with measurements and no knowledge of the ground truth

# make a guess of the initial start point
xfk= np.array([0,1,1,0])
Pfk= np.array([[5**2,0,0,0],
               [0,5**2,0,0],
               [0,0,2**2,0],
               [0,0,0,2**2]]) 


# save before measurement update
Xfk_prior=np.zeros((N,dim))
Covfk_prior=np.zeros((N,dim,dim))

Xfk_prior[0]=xfk
Covfk_prior[0]=Pfk

# save after measurement update
Xfk=np.zeros((N,dim))
Covfk=np.zeros((N,dim,dim))

Xfk[0]=xfk
Covfk[0]=Pfk

# assuming we have no idea what control was used on the robot
uk=np.array([0])  

# fig=plt.figure("Estimated trajectory")
# ax = fig.add_subplot(111)

fig2=plt.figure("Estimated trajectory with covariance")
ax2 = fig2.add_subplot(111)

# now run as if we are in real-time 
for i in range(1,N):
    xfk1, Pfk1 = propagate(dt*i, dt,xfk,Pfk,uk,Ak,Bk,Qk)
    
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
    
    
    plt.show()
    plt.pause(1)
    