
# %% logging
import loggerconfig as logconf
logger = logconf.getLogger(__name__)

logger.info('Info log message')
logger.debug('debug message')
logger.error('error example')
logger.verbose('verbose log message')
logger.warning('warn message')
logger.critical('critical message')

# %% imports
import os
import threading
import pickle as pkl
from random import shuffle
import matplotlib
# try:
#     matplotlib.use('TkAgg')
# except:
# matplotlib.use('nbAgg')
import matplotlib.pyplot as plt
colmap = plt.get_cmap('gist_rainbow')

import os
from matplotlib import cm
from scipy.linalg import block_diag
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import numpy as np
import numpy.linalg as nplg
from uq.gmm import gmmbase as uqgmmbase
from uq.gmm import merger as uqgmmmerg
from uq.gmm import splitter as uqgmmsplit
from physmodels import motionmodels as phymm
from physmodels import duffing as phyduff
from physmodels import sensormodels as physm
from physmodels import sensormodels_fov as physmfov
from uq.uqutils import recorder as uqrecorder
from physmodels import targets as phytarg
from uq.uqutils import metrics as uqmetrics
from utils.math import geometry as utmthgeom
from uq.uqutils import simmanager as uqsimmanager
from utils.plotting import geometryshapes as utpltshp
from utils.plotting import surface as utpltsurf
from uq.quadratures import cubatures as quadcub
from uq.gmm import merger as uqgmmmerg
from uq.gmm import splitter as uqgmmsplit   
import uq.quadratures.cubatures as uqcb
import uq.filters.kalmanfilter as uqkf
from uq.filters import sigmafilter as uqfsigf
from uq.filters import gmm as uqgmmf
from sklearn import mixture
import scipy.optimize as scopt #block_diag

import collections as clc


# %% script-level properties

runfilename = __file__
metalog="""
idea of meas updatr
Author: Venkat
Date: June 4 2020

meas update for gmm
"""


simmanger = uqsimmanager.SimManager(t0=0,tf=2,dt=0.1,dtplot=0.1,
                                  simname="GMM-NEW-UPDATE",savepath="simulations",
                                  workdir=os.getcwd())

simmanger.initialize()

# %%
m0 = np.array([5,5])
P0 = 0.01*np.identity(2)
X0=np.random.multivariate_normal(m0,P0, 10000 )


duffmodel = phyduff.Duffing()
X=duffmodel.integrate(simmanger.tvec,m0)

Xbatch = duffmodel.integrate_batch(simmanger.tvec,X0)
# X1=duffmodel.propforward( simmanger.tvec[0], simmanger.tvec[-1], X0[0,:], uk=0)

fig = plt.figure("Duff")
ax = fig.add_subplot(111,label='contour')

ax.set_title("duffing 1")
ax.plot(X[:,0],X[:,1])

# fig = plt.figure("Duff-MC")
# ax = fig.add_subplot(111,label='contour')
# for i in range(simmanger.ntimesteps):
#     ax.cla()
#     ax.plot(X[:,0],X[:,1],'r')
#     ax.plot(Xbatch[:,i,0],Xbatch[:,i,1],'bo')
#     plt.pause(1)


# %% measurement update
plt.close('all')
R = block_diag((5)**2, (5*np.pi/180)**2)
sensormodel = physm.Disc2DRthetaSensorModel(R, recorderobj=None, recordSensorState=False)

splitterConfig=uqgmmsplit.splitterConfigDefault
splitterConfig.rsfac = 1
splitterConfig.sigL =2
splitterConfig.wtL = 0.5
splitterConfig.Ngh = 5
gmm = uqgmmsplit.splitGaussianND(m0,P0,splitterConfig=splitterConfig)

modefilterer=uqfsigf.Sigmafilterer( sigmamethod=quadcub.SigmaMethod.UT)
gmmfilterer = uqgmmf.GMMfilterer(modefilterer)

t=simmanger.tvec[0]
dt=simmanger.tvec[-1]-simmanger.tvec[0]

gmmfk = gmmfilterer.propagate(t, dt,duffmodel,gmm,None,inplace=False)

xktruth = Xbatch[3000,-1,:]
zk,_,_ = sensormodel(t+dt, dt, xktruth)
gmmuk, gmmz,Pxzf, Lj,likez = gmmfilterer.measUpdate(t+dt, dt,gmmfk,sensormodel,zk,inplace=False)



fig = plt.figure("Duff")
ax = fig.add_subplot(111,label='contour')
ax.set_title("duffing 1")
ax.plot(X[:,0],X[:,1])
XX = uqgmmbase.plotGMM2Dcontour(gmm,nsig=1,N=100,rettype='list')
for cc in range(gmm.Ncomp):
    ax.plot(XX[cc][:,0],XX[cc][:,1],'r')
XX = uqgmmbase.plotGMM2Dcontour(gmmfk,nsig=1,N=100,rettype='list')
for cc in range(gmm.Ncomp):
    ax.plot(XX[cc][:,0],XX[cc][:,1],'b')
ax.plot(Xbatch[:,-1,0],Xbatch[:,-1,1],'b.')        
ax.plot(Xbatch[:,0,0],Xbatch[:,0,1],'r.')  
ax.plot(xktruth[0],xktruth[1],'k*')  


fig = plt.figure("MainSim-Surf-0")
ax = fig.add_subplot(111, projection='3d')
ax.plot([xktruth[0]],[xktruth[1]],'k*') 
ax.plot(X[:,0],X[:,1])
xx,yy,p = uqgmmbase.plotGMM2Dsurf(gmm,Ng=50)
ax.plot_surface(xx,yy,p,color='b',alpha=0.6,linewidth=1) 

fig = plt.figure("MainSim-Surf prior")
ax = fig.add_subplot(111, projection='3d')
ax.plot([xktruth[0]],[xktruth[1]],'k*') 
ax.plot(X[:,0],X[:,1])
xx,yy,p = uqgmmbase.plotGMM2Dsurf(gmmfk,Ng=50)
ax.plot_surface(xx,yy,p,color='r',alpha=0.6,linewidth=1) 

fig = plt.figure("MainSim-Surf post")
ax = fig.add_subplot(111, projection='3d')
ax.plot([xktruth[0]],[xktruth[1]],'k*') 
ax.plot(X[:,0],X[:,1])
xx,yy,p = uqgmmbase.plotGMM2Dsurf(gmmuk,Ng=50)
ax.plot_surface(xx,yy,p,color='g',alpha=0.6,linewidth=1) 

# %%
m,P = gmmuk.meanCov()
gmmukcopy = gmmuk.makeCopy()


def confunc(w):
   gmmukcopy.setwts(w)
   m1,P1 = gmmukcopy.meanCov()
   return nplg.norm(m-m1)

c1=scopt.NonlinearConstraint(confunc, 0, 0.05)
A=np.ones(gmmuk.Ncomp)
unitypartconstraint = scopt.LinearConstraint(A, 0.99, 1.01)
        
def func(w):
   gmmukcopy.setwts(w)
   X,w=gmmukcopy.generateUTpts()
   lnX = np.log(gmmukcopy.pdf(X))
   H = -np.sum(lnX.dot(w))
   
   return H

bounds=[(0,1)]*gmmuk.Ncomp
x0 = gmmuk.wts.copy()
res = scopt.minimize(func, x0, constraints=[c1,unitypartconstraint],bounds=bounds)
 
gmmukcopy.setwts(res.x)      
fig = plt.figure("MainSim-Surf post-Updated")
ax = fig.add_subplot(111, projection='3d')
ax.plot([xktruth[0]],[xktruth[1]],'k*') 
ax.plot(X[:,0],X[:,1])
xx,yy,p = uqgmmbase.plotGMM2Dsurf(gmmukcopy,Ng=50)
ax.plot_surface(xx,yy,p,color='g',alpha=0.6,linewidth=1) 
    