
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
import collections as clc
plt.close('all')
plt.ion()
from sklearn import mixture
import time
import utils.timers as utltm
from robot import templatetraj as robttj
from robot import gridrobot as robtgr
from maps import gridmap
import copy
import robot.filters.robot2Dfilters as rbf2df
from sensortasking import robotdp as robdb
import pickle as pkl
import numpy as np
import numpy.linalg as nplnalg
import scipy.linalg as sclnalg
from uq.information import distance as uqinfodis
import robot.filters.robot2Dfilters as rbf2df
import collections as clc
from uq.gmm import gmmbase as uqgmmbase
import utils.timers as utltm
from random import shuffle
import multiprocessing as mp
import matplotlib.pyplot as plt
from utils.math import geometry as utmthgeom
import pickle as pkl
# %% script-level properties
fname = "debug-info-tvec-8.pkl"
with open(fname,'rb') as FF:
    rid,ridstates,robots,targetset,tvec,Targetfilterer,splitterConfig,mergerConfig = pkl.load(FF)

# %%
targetset[0].recorderprior
targetset[0].recorderprior.data['t']
targetset[0].recorderpost.data['t']

# %%
colmap = plt.get_cmap('gist_rainbow')
NUM_COLORS = 20
colors = [colmap(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
shuffle(colors)

def plotinfomap(targetset,robots,infocost,ridstates,t):
    
    fig = plt.figure("Info-Surf: %f"%t)

    

    
    ax_list = fig.axes
    if len(ax_list)==0:     
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = ax_list[0]
    ax.cla()
    
    for i in range(targetset.ntargs):
        # xktruth = targetset[i].groundtruthrecorder.getvar_uptotime_stacked('xtk',t)
        gmmu = targetset[i].recorderprior.getvar_bytime('gmmfk',t)
        # if gmmu is None:
        #     gmmu = targetset[i].recorderprior.getvar_bytime('gmmfk',t+dt)
            
        gmmupos = uqgmmbase.marginalizeGMM(gmmu,targetset[i].posstates)
        XX = uqgmmbase.plotGMM2Dcontour(gmmupos,nsig=1,N=100,rettype='list')
        for cc in range(gmmupos.Ncomp):
            ax.plot(XX[cc][:,0],XX[cc][:,1],c=colors[i])
            # ax.annotate("wt: "+str(gmmupos.w(cc))[:5],gmmupos.m(cc)[0:2],gmmupos.m(cc)[0:2]-3,color=colors[i],fontsize='x-small')
        # ax.plot(xktruth[:,0],xktruth[:,1],linestyle='--',c=colors[i])
        # ax.plot(xktruth[-1,0],xktruth[-1,1],c=colors[i],marker='*')
        
        
    robots[0].mapobj.plotmap(ax)
    # for r in robots:
    #     r.plotrobot(ax)
            
    p=np.zeros(ridstates.shape[0])
    x=np.zeros(ridstates.shape[0])
    y=np.zeros(ridstates.shape[0])
    for n in range(ridstates.shape[0]):
        x[n] = ridstates[n,0]
        y[n] = ridstates[n,1]
        p[n] = infocost[t][n] 
    ax.scatter(x, y, p, c='b',marker='o')
    plt.pause(1)
    plt.ion()
    # plt.show(block=True)
    plt.show()
    plt.pause(0.1)


# %%
Nmc = 10000
i=1
gmmfk = targetset[i].gmmfk.makeCopy()
# mu = np.zeros(5)
# P0 = 0.1**2*np.identity(5)
# gmmfk = uqgmmbase.GMM(None,None,None,0)
# gmmfk.appendCompArray(np.array([mu]),np.array([P0]),np.array([1]),renormalize = True)

Xmc = gmmfk.random(Nmc)
# Xmc,_ = quadcub.GH_points(mu,P0,5)
pt = gmmfk.pdf(Xmc)
clf = mixture.GaussianMixture(n_components=5, covariance_type='full')
clf.fit(Xmc)
gmmuk = uqgmmbase.GMM(None,None,None,0)
gmmuk.appendCompArray(clf.means_,clf.covariances_,clf.weights_,renormalize = True)
wtsopt=uqgmmbase.optimizeWts(gmmuk,Xmc,pt)
wtsopt=wtsopt/np.sum(wtsopt)
gmmuk2=gmmuk.makeCopy()
gmmuk2.wts = wtsopt
pt2 = gmmuk2.pdf(Xmc)
dd1=uqinfodis.hellingerDistGMM(gmmfk,gmmuk)
dd2=uqinfodis.hellingerDistGMM(gmmfk,gmmuk2)
dd3=uqinfodis.hellingerDistGMM(gmmuk,gmmuk2)
print(dd1)
print(dd2)
print(dd3)
fig = plt.figure("original-surf")
ax = fig.add_subplot(111, projection='3d')
xx,yy,p = uqgmmbase.plotGMM2Dsurf(uqgmmbase.marginalizeGMM(gmmfk,[0,1]),Ng=50)
ax.plot_surface(xx,yy,p,alpha=0.7,linewidth=1)
xx,yy,p = uqgmmbase.plotGMM2Dsurf(uqgmmbase.marginalizeGMM(gmmuk2,[0,1]),Ng=50)
ax.plot_surface(xx,yy,p,alpha=0.7,linewidth=1)

ax.set_xlabel('x')
ax.set_ylabel('y')

dd=100*np.abs(pt-pt2)/pt
print(np.percentile(dd,75))

# %%
class InfoConfig:
    def __init__(self):
        self.bins = (tuple(np.linspace(0,100,75)),
                     tuple(np.linspace(0,100,75)),
                     tuple(np.linspace(-5,5,20)),
                     tuple(np.linspace(-5,5,20)),
                     tuple(np.linspace(-1,1,20)))
        self.binmaxwidth = [3,3,1,1,0.5]
        
infoconfig = InfoConfig()

for i in range(targetset.ntargs):
    targetset[i].freezeState()
for r in range(rid+1):
    robots[r].freezeState()    

D=clc.defaultdict(dict)
for n in range(ridstates.shape[0]):
    D[tvec[0]][n]=0
    
for j in range(0,1):
    # for n in range(ridstates.shape[0]):
        # with utltm.TimingContext("robot pos: "):
    n=45  # 53 45 63
    It=0
    robots[rid].xk[0:2] = ridstates[n] 
    robots[rid].updateSensorModel()
    for r in range(rid):
        robots[r].xk = robots[r].statehistory[tvec[j]]
        robots[r].updateSensorModel()
        
    for i in range(targetset.ntargs):
        dt = tvec[j]-tvec[j-1]
        # gmmfk = targetset[i].recorderprior.getvar_bytime('gmmfk',tvec[j])
        targetset[i].setStateFromPrior(tvec[j],['gmmfk','xfk','Pfk'])
        
        gmmfkpos = uqgmmbase.marginalizeGMM(targetset[i].gmmfk,targetset[i].posstates)
        Nz = 1000
        X = gmmfkpos.random(Nz)
        w = np.ones(Nz)/Nz
        # X,w = gmmfkpos.generateUTpts()
        
        
        I=[]
        cache={}
        for s in range(X.shape[0]):
            Zk=clc.defaultdict(list)
            for r in range(rid+1):
                zij,isinsidek,Lk = robots[r].sensormodel.generateRndMeas(tvec[j], dt, X[s],useRecord=False)     
                if isinsidek == 0:
                    Zk[i].append(None)
                else:
                    Zk[i].append(zij)
            

            ykey = tuple([1 if zz is not None else 0 for zz in Zk[i]])
            if ykey not in cache:
                # measUpdateFOV_PFGMM
                # measUpdateFOV_randomsplitter
                # kk = tuple([1 if zz is not None else 0 for zz in Zk[i]])
                # if kk not in cache:
                print(Zk[i])
                splitterConfig.NcompIN = 5
                splitterConfig.NcompOUT = 5
                mergerConfig.fixedComps = 5
                gmmuk=rbf2df.measUpdateFOV_PFGMM(tvec[j],dt,robots[:(rid+1)],targetset[i],Zk[i],Targetfilterer,infoconfig,
                                                updttarget=False,
                                                splitterConfig = splitterConfig,
                                                mergerConfig = mergerConfig,
                                                computePriorPostDist=True)
               
                fig = plt.figure("Debug Info-Surf: %f"%tvec[j])
                ax_list = fig.axes
                if len(ax_list)==0:     
                    ax = fig.add_subplot(111) #, projection='3d'
                else:
                    ax = ax_list[0]
                ax.cla()
                ax.plot(X[s,0],X[s,1],'k*')                    
                gmmfkpos = uqgmmbase.marginalizeGMM(targetset[i].gmmfk,targetset[i].posstates)
                gmmukpos = uqgmmbase.marginalizeGMM(gmmuk,targetset[i].posstates)
                XX = uqgmmbase.plotGMM2Dcontour(gmmfkpos,nsig=2,N=100,rettype='list')
                for cc in range(gmmfkpos.Ncomp):
                    ax.plot(XX[cc][:,0],XX[cc][:,1],c=colors[i])
                XX = uqgmmbase.plotGMM2Dcontour(gmmukpos,nsig=2,N=100,rettype='list')
                for cc in range(gmmukpos.Ncomp):
                    ax.plot(XX[cc][:,0],XX[cc][:,1],c=colors[i],linestyle='--')
                
                robots[0].mapobj.plotmap(ax)
                for r in range(rid+1):
                    robots[r].plotrobot(ax)  
                    
                # dd=uqinfodis.hellingerDistGMM(targetset[i].gmmfk,gmmuk)
                dd = targetset[i].context[tvec[j]]['info']
                cache[ykey] = dd
                print(dd)
                ax.set_title("Info = %f, target %d, time: %f"%(dd,i,tvec[j]))
                plt.pause(1)
                plt.ion()
                plt.show()
                plt.pause(0.5)
            else:
                dd = cache[ykey]
            # I=I+w[s]*dd
                I.append(dd)
        I=np.min(I)
            
        print("Info I = ",I, " target = ",i )
        It = It + I
    print("total info = ",It)
    #     print("Done info grid for %d/%d at timestep %d/%d"%(n,ridstates.shape[0],j,len(tvec)))
    #     D[tvec[j]][n] = It
        
    # plotinfomap(targetset,robots[:(rid+1)],D,ridstates,tvec[j])
    # break

# for i in range(targetset.ntargs):
#     targetset[i].defrostState()
# for r in range(rid+1):
#     robots[r].freezeState()  
#     robots[r].updateSensorModel()
        