# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 08:19:18 2021

@author: Nagnanamus
"""
import numpy as np
from numpy.linalg import multi_dot
import numpy.linalg as nplg
import numpy.linalg as nplinalg

import scipy.linalg as sclg
import matplotlib.pyplot as plt
import scipy.optimize as scopt
import pdb
import numba
from numba import vectorize, float64,guvectorize,int64,double,int32,float32
from numba import njit, prange,jit

from lidarprocessing import turtlebotModels as tbotmd
plt.close("all")



#%% Kalman Filter



fmodel = tbotmd.CAUMmodel_1()




#%% generate a ground truth i.e. measurements
plt.close("all")
N=200
xt=np.array([0,0,0,0,0,0,0,0],dtype=np.float64)
Xt=np.zeros((N,dim),dtype=np.float64)
Zk=np.zeros((N,hn),dtype=np.float64)
Xt[0]=xt

vidx=[2,3]
aidx=[4,5]
omidx = 7

for i in range(1,N):
    Xt[i]=fmodel(dt,Xt[i-1])
    


    s=np.random.rand(1)[0]
    if s>0.8:
        Omegadt = 2*np.random.rand(1)[0]-1
        Omegadt = clipvalue(Omegadt,0.05,-0.05)
        Xt[i,omidx] = Xt[i,omidx]+ Omegadt
        
    if s>0.6:
        adt = 2*np.random.rand(1)[0]-1        
        adt = clipvalue(adt,0.07,-0.07)        
        Xt[i,aidx[0]] = adt+Xt[i,aidx[0]]
    
    if s>0.6:
        adt = 2*np.random.rand(1)[0]-1        
        adt = clipvalue(adt,0.07,-0.07)        
        Xt[i,aidx[1]] = adt+Xt[i,aidx[1]]
        
        
    # geenrate some random measurements
    Zk[i]=hmodel(dt,Xt[i])+np.matmul(sclg.sqrtm(Rk), np.random.randn(hn))
    # Zk[i]=CACTmeasmodel_1_lidar(dt,Xt[i])+np.matmul(sclg.sqrtm(Rk), np.random.randn(hn))
        
        

plt.figure()
plt.plot(Xt[:,0],Xt[:,1],'k*-')  
# plt.plot(Zk[1:,0],Zk[1:,1],'r*')
        
# pure integration        
#%%    Now run Kalman filter only with measurements and no knowledge of the ground truth
plt.close("all")
# make a guess of the initial start point
# xfk= x0 = np.array([0,0,0,0,0,0],dtype=np.float64)
# Pfk= P0 = np.array([[0.5**2,0,0,0,0,0],
#                [0,0.5**2,0,0,0,0],
#                [0,0,0.1**2,0,0,0],
#                [0,0,0,0.2**2,0,0],
#                [0,0,0,0,0.05**2,0],
#                [0,0,0,0,0,0.05**2]],dtype=np.float64) 

xfk= x0 = np.array([0,0,0,0,0,0,0,0],dtype=np.float64)
Pfk= P0 = np.diag([0.1**2,0.1**2,0.01**2,0.01**2,0.001**2,0.001**2,0.001**2,0.001**2])



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
    
    
    Xfk_prior[i]=xfk1
    Covfk_prior[i]=Pfk1
    
    # do the measurmenet update with the measurement at this time
    zk = Zk[i]
    # hk=CACTmeasmodel_1_lidar
    # Hk=CACTmeasmodeljac_1_lidar(dt, xfk1)
    
    hk=hmodel
    Hk=hmodeljac(dt, xfk1)
    xu, Pu = measUpdate(i*dt, dt,xfk1,Pfk1,hmodel,Hk,Rk,zk)

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
    ax2.plot(Xfk[:i+1,0],Xfk[:i+1,1],'b.-',label="estimate")
    
    # Xe= getCovEllipsePoints2D(Xfk_prior[i][0:2],Covfk_prior[i][0:2,0:2],nsig=1,N=100)
    # ax2.plot(Xe[:,0],Xe[:,1],'g--')
    
    # Xe= getCovEllipsePoints2D(Xfk[i][0:2],Covfk[i][0:2,0:2],nsig=1,N=100)
    # ax2.plot(Xe[:,0],Xe[:,1],'g')
    
    # ax2.set_xlim(np.min(Xt[:,0])-10,np.max(Xt[:,0])+10)
    # ax2.set_ylim(np.min(Xt[:,1])-10,np.max(Xt[:,1])+10)
    # ax2.legend()
    


    plt.show()
    plt.pause(0.2)
    
# fig3=plt.figure("Estimated Turn-rate")
# ax3 = fig3.add_subplot(111)
# ax3.plot(dt*np.arange(0,N),Xt[:,4],label="True ground truth turn rate ")
# ax3.plot(dt*np.arange(0,N),Xfk[:,4],label="Estimated turn rate ")
# ax3.plot(dt*np.arange(0,N),Xfkbatch[:,4],label="Batch Estimated turn rate ")
# ax3.legend()
# plt.show()


