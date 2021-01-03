"""IMM+JPDA main function for testinf and simulation."""

import matplotlib
import numpy as np
import numpy.linalg as nplg
from scipy.linalg import block_diag
import collections as clc
import os
import pandas as pd
from uq.motfilter import jpda
import uq.filters.kalmanfilter as uqkf
from uq.filters import sigmafilter as uqsigf
from uq.uqutils.random import genRandomMeanCov
from physmodels import motionmodels as phymm
from physmodels import sensormodels as physm
from physmodels import targets as phytarg
from uq.filters import imm as immfilter
from uq.gmm import gmmbase as uqfgmmbase
from uq.uqutils import recorder as uqrecorder
from uq.uqutils import metrics as uqmetrics
from uq.uqutils import simmanager as uqsimmanager
from uq.quadratures import cubatures as quadcub
import uq.quadratures.cubatures as uqcb
import MOC.moctreepf as moctrpf
try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from loggerconfig import *

import scipy.optimize as scopt #block_diag
from uq.transformers import transforms as uqtransf
import collections as clc
from uq.stats import moments as uqstat
import physmodels.twobody.constants as tbpconst
from physmodels.twobody import tbpcoords
from ndpoly import polybasis as plybss
earthconst = tbpconst.PlanetConstants()
tbpdimconvert = tbpcoords.DimensionalConverter(earthconst)


plt.close('all')
runfilename = __file__

#%%
logger = logging.getLogger(__name__)

logger.info('Info log message')
logger.debug('debug message')

logger.error('error example')
logger.verbose('verbose log message')
# try:
#     raise Exception('exception message')
# except:
#     logger.exception('error occured')


logger.debug('debug message')
logger.info('info message')
logger.warning('warn message')
logger.error('error message')
logger.critical('critical message')

#%%
metalog="""
ACC 2021 MOC-PF paper simulations
Author: Venkat
Date: June 4 2020


"""
t0 = tbpdimconvert.true2can_time(0)
tf = tbpdimconvert.true2can_time(300*60*60)
dt = tbpdimconvert.true2can_time(36*60*60)

simdata['satprob']={'dt':dt,'t0':t0,'tf':tf}

simmanger = uqsimmanager.SimManager(t0=t0,tf=tf,dt=dt,dtplot=0.1,
                                  simname="ACC-2021-MOC-PF",savepath="simulations",
                                  workdir=os.getcwd())

simmanger.initialize()




#%%
def gaussianPDF(X,m,P):
    Pinv = nplalg.inv(P)
    a = X-m
    c = 1/np.sqrt(nplalg.det(2*np.pi*P))
    if X.ndim==1:
        y = c*np.exp(-0.5*multi_dot([a,Pinv,a]))
    else:
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y[i] = np.exp(-0.5*multi_dot([a[i],Pinv,a[i]]))
        y = c*y
    return y

def gaussianPDF1D(X,m,var):
    Pinv = 1/var
    a = X-m
    c = 1/np.sqrt(2*np.pi*var)
    y = c*np.exp(-0.5*Pinv*a**2)
    return y


moctrpf

# eXAMPLE 2: polar to cartesian

def hfunc(x):
    p0=np.array([40,40])
    if x.ndim>1:
        r=nplg.norm( x-p0,axis=1)
    else:
        r= nplg.norm(x-p0)
    return r

def problike(z,x):
    # single meas z and multiple x
    var = 5**2
    return gaussianPDF1D(z-hfunc(x),0,var)
    
def func2(x):
    if x.ndim==1:
        r=x[0]
        th=x[1]
        xk1 = r*np.array([np.cos(th),np.sin(th)])
        return xk1
    else:
        xk1 = np.vstack([x[:,0]*np.cos(x[:,1]),x[:,0]*np.sin(x[:,1])]).T
        return xk1
    
def func2jac(x):
    if x.ndim==1:
        r=x[0]
        th=x[1]
        jac = np.array([[np.cos(th),-r*np.sin(th)],[np.sin(th),r*np.cos(th)]])
        return jac
    else:
        jac=[]
        for i in range(x.shape[0]):
            r=x[i,0]
            th=x[i,1]
            jac.append( np.array([[np.cos(th),-r*np.sin(th)],[np.sin(th),r*np.cos(th)]]) )

        return jac


x0eg2 = np.array([30,np.pi/2])
P0eg2 = np.array([[2**2,0],[0,(15*np.pi/180)**2]])
Nmc = 10000
Xmc0=np.random.multivariate_normal(x0eg2,P0eg2, Nmc )
pmc0 = multivariate_normal(mean=x0eg2,cov=P0eg2).pdf(Xmc0)
Xmcprop = func2(Xmc0)
jacprop = func2jac(Xmc0)
pmcprop = np.array( [pmc0[i]/nplg.det(jacprop[i]) for i in range(Nmc) ] )

fig = plt.figure("MC before")
ax = fig.add_subplot(111)
ax.cla()
ax.plot(Xmc0[:,0],Xmc0[:,1],'r.')
fig = plt.figure("MC after")
ax = fig.add_subplot(111)
ax.cla()
ax.plot(Xmcprop[:,0],Xmcprop[:,1],'b.')



# Algorithms - using MC
"""
- use initial MC-pf to estimate mean and cov
- prior:
    - build tree with depth d=3 centered at mu, 6sig
    - iterate boxes with var>thresh_var
        - get more points for this box



"""

# %% IMM weights plots

mpfk = jpdamotlist['UKFIMMJPDA'].targetset[i].recorderpost.getvar_alltimes('modelprobfk')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(simmanger.tvec, mpfk[:,0], label='UM')
ax.plot(simmanger.tvec, mpfk[:,1], label='CT')
ax.legend()
plt.pause(0.1)
simmanger.savefigure(fig, ['post'], 'UKFIMMJPDA'+'immwts.png')

# %% Saving
simmanger.finalize()

simmanger.save(metalog, mainfile=runfilename, simdata=simdata )

simmanger.summarize()
# debugStatuslist = jpdamot.debugStatus()
# uqutilhelp.DebugStatus().writestatus(debugStatuslist,simmanger.debugstatusfilepath)
