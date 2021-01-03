
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


import numpy as np
import robot.gridrobot as robgrid
from uq.information import distance as uqinfodis
import collections as clc
import copy
from utils.math import geometry as utmthgeom
from uq.gmm import gmmbase as uqgmmbase
from sensortasking import robotinfo as sensrbinfo
# %% script-level properties


class InfoConfig:
    def __init__(self):
        self.bins = (tuple(np.linspace(0,100,30)),
                     tuple(np.linspace(0,100,30)),
                     tuple(np.linspace(-5,5,20)),
                     tuple(np.linspace(-5,5,20)),
                     tuple(np.linspace(-1,1,20)))
        self.binmaxwidth = [3,3,1,1,0.5]

infoconfigDefault = InfoConfig()

def plotinfomap(targetset,robots,infocost,ridstates,t):
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure("Info-Surf: %f"%t)
    print("working")
    colmap = plt.get_cmap('gist_rainbow')
    NUM_COLORS = 20
    colors = [colmap(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    shuffle(colors)

    
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
        XX = uqgmmbase.plotGMM2Dcontour(gmmupos,nsig=2,N=100,rettype='list')
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
    # k = f'{t:06.2f}'.replace('.','-')
    # simmanager.savefigure(fig, ['SimSnapshot', 'InfoMap'], k+'.png',data=[targetset,robots,infocost,ridstates,t])

def getinfotimestep(j,rid,ridstates,robots,targetset,tvec,Targetfilterer,infoconfig,splitterConfig,mergerConfig):
    D={}
    for n in range(ridstates.shape[0]):
        # with utltm.TimingContext("robot pos: "):
        
        robots[rid].xk[0:2] = ridstates[n] 
        robots[rid].updateSensorModel()
        for r in range(rid):
            robots[r].xk = robots[r].statehistory[tvec[j]]
            robots[r].updateSensorModel()
        
        It=[]   
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
                
                # measUpdateFOV_PFGMM
                # measUpdateFOV_randomsplitter
                ykey = tuple([1 if zz is not None else 0 for zz in Zk[i]])
                if ykey not in cache:
                    dm = mergerConfig.doMerge
                    mergerConfig.doMerge = False
                    gmmuk = rbf2df.measUpdateFOV_randomsplitter(tvec[j],dt,robots[:(rid+1)],targetset[i],Zk[i],Targetfilterer,infoconfig,
                                                    updttarget=False,
                                                    splitterConfig = splitterConfig,
                                                    mergerConfig = mergerConfig,
                                                    computePriorPostDist=True)
                    mergerConfig.doMerge = dm
                    

                    dd = targetset[i].context[tvec[j]]['info']
                    cache[ykey] = dd
                else:
                    dd = cache[ykey]
                    
                # I=I+w[s]*dd
                I.append(dd)
            
            I = np.min(I)
            
            It.append( I )
        
        print("Done info grid for %d/%d at timestep %d/%d"%(n,ridstates.shape[0],j,len(tvec)))
        D[n] = It[0]

    return D

def robotTargetInfo(rid,ridstates,robots,targetset,tvec,Targetfilterer,infoconfig,splitterConfig,mergerConfig):
   
    for i in range(targetset.ntargs):
        targetset[i].freezeState()
    for r in range(rid+1):
        robots[r].freezeState()    

    D=clc.defaultdict(dict)
    for n in range(ridstates.shape[0]):
        D[tvec[0]][n]=0
    jobs = {} 
    for j in range(1,len(tvec)):
        args = (j,rid,ridstates,robots,targetset,tvec,Targetfilterer,infoconfig,splitterConfig,mergerConfig)
        # jobs[tvec[j]]= utrq.redisQ.enqueue(getinfotimestep,args=args,result_ttl=86400, job_timeout=1000) 
        d = getinfotimestep(*args)
        D[tvec[j]] = d
        plotinfomap(targetset,robots[:(rid+1)],D,ridstates,tvec[j])
           
    for i in range(targetset.ntargs):
        targetset[i].defrostState()
    for r in range(rid+1):
        robots[r].defrostState()  
        robots[r].updateSensorModel()
        
    return D

# %%
folder= "simulations/DynamicSensorTasking-DP-Template_2020-07-14-12H-40M-52s/debugdata/dynamicProgRobot2Dgrid_SeqRobot"
fname = "24.pkl"
with open(os.path.join(folder,fname),'rb') as FF:
    tvec,robots,targetset,Targetfilterer,infoconfig,splitterConfig,mergerConfig = pkl.load(FF)

tvec = tvec[0:2]
Ukfilter = [None]*targetset.ntargs
InfoCosts=[]
Cost2gos=[]

# propagate targets by one step
for j in range(len(tvec)-1):
    for i in range(targetset.ntargs):
        Targetfilterer.propagate(tvec[j],tvec[j+1]-tvec[j],
                                      targetset[i],
                                      Ukfilter[i],
                                      updttarget=True)

# save the initial states of the robots
for r in range(len(robots)):
    for j in range(len(tvec)):
        robots[r].statehistory.pop(tvec[j],None)
        robots[r].controllerhistory.pop(tvec[j],None)
    robots[r].statehistory[tvec[0]] = robots[r].xk.copy()
    
# doing DP for robots in seq and from end time
for r in range(len(robots)):
    ridstates = robots[r].mapobj.XYgvec
    infocost = robotTargetInfo(r,ridstates,robots,targetset,tvec,Targetfilterer,infoconfig,splitterConfig,mergerConfig)



# %%

NUM_COLORS = 20
colors = [colmap(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
shuffle(colors)
def plotgmmtarget(fig,ax,gmmus,gmmfs,xktruths,t,robots,posstates):
    plt.figure(fig.number)
    ax.cla()
    ax.set_title("time step = %f"%(t,) )
    robots[0].mapobj.plotmap(ax)
    for r in robots:
        r.plotrobot(ax)
        
    # xktruth = targetset[i].groundtruthrecorder.getvar_uptotime_stacked('xtk',t+dt)
    for i,gmm in enumerate(gmmfs):
        gmmupos = uqgmmbase.marginalizeGMM(gmm,posstates)
        XX = uqgmmbase.plotGMM2Dcontour(gmmupos,nsig=2,N=100,rettype='list')
        for cc in range(gmmupos.Ncomp):
            ax.plot(XX[cc][:,0],XX[cc][:,1],c=colors[i])
            ax.annotate("wt: "+str(gmmupos.w(cc))[:5],gmmupos.m(cc)[0:2],gmmupos.m(cc)[0:2]-3,color=colors[i],fontsize='x-small')
    
    for i,gmm in enumerate(gmmus):
        gmmupos = uqgmmbase.marginalizeGMM(gmm,posstates)
        XX = uqgmmbase.plotGMM2Dcontour(gmmupos,nsig=2,N=100,rettype='list')
        for cc in range(gmmupos.Ncomp):
            ax.plot(XX[cc][:,0]+1,XX[cc][:,1]+1,linewidth=2,linestyle='--',c=colors[i])
            ax.annotate("wt: "+str(gmmupos.w(cc))[:5],gmmupos.m(cc)[0:2],gmmupos.m(cc)[0:2]-3,color=colors[i],fontsize='x-small')
            
    for i,xktruth in enumerate(xktruths):
        ax.plot(xktruth[-1,0],xktruth[-1,1],c=colors[i],marker='*')
    # ax.annotate(targetset[i].targetName,xktruth[-1,0:2],xktruth[-1,0:2]+2,color=colors[i],fontsize='x-small')

# 
    ax.axis('equal')
    ax.set(xlim=(-15, 115), ylim=(-15, 115))
    plt.pause(0.1)
    plt.ion()
    plt.show()
    

rid = 0
n = 70
robots[rid].xk[0:2] = ridstates[n] 
robots[rid].updateSensorModel()
for r in range(rid):
    robots[r].xk = robots[r].statehistory[tvec[j]]
    robots[r].updateSensorModel()

It=0   
gmmfs=[]
gmmus=[] 
for i in range(targetset.ntargs):
    dt = tvec[j]-tvec[j-1]
    # gmmfk = targetset[i].recorderprior.getvar_bytime('gmmfk',tvec[j])
    targetset[i].setStateFromPrior(tvec[j],['gmmfk','xfk','Pfk'])
    
    gmmf = targetset[i].gmmfk.makeCopy()
    # xktruth = targetset[i].groundtruthrecorder.getvar_uptotime_stacked('xtk',tvec[j])
    
    gmmfkpos = uqgmmbase.marginalizeGMM(targetset[i].gmmfk,targetset[i].posstates)
    Nz = 1000
    X = gmmfkpos.random(Nz)
    w = np.ones(Nz)/Nz
    # X,w = gmmfkpos.generateUTpts()

    
    I=[]
    cache={}
    cnts={}
    for s in range(X.shape[0]):
        Zk=clc.defaultdict(list)
        for r in range(rid+1):
            zij,isinsidek,Lk = robots[r].sensormodel.generateRndMeas(tvec[j], dt, X[s],useRecord=False)     
            if isinsidek == 0:
                Zk[i].append(None)
            else:
                Zk[i].append(zij)
        
        # measUpdateFOV_PFGMM
        # measUpdateFOV_randomsplitter
        ykey = tuple([1 if zz is not None else 0 for zz in Zk[i]])
        if ykey not in cnts:
            cnts[ykey]=1
        else:
            cnts[ykey]=cnts[ykey]+1
            
        if ykey not in cache:
            dm = mergerConfig.doMerge
            mergerConfig.doMerge = False
            gmmuk = rbf2df.measUpdateFOV_randomsplitter(tvec[j],dt,robots[:(rid+1)],targetset[i],Zk[i],Targetfilterer,infoconfig,
                                            updttarget=False,
                                            splitterConfig = splitterConfig,
                                            mergerConfig = mergerConfig,
                                            computePriorPostDist=True)
            mergerConfig.doMerge = dm
            

            dd = targetset[i].context[tvec[j]]['info']
            cache[ykey] = dd
            
            fig1 = plt.figure("Debug plot target %d,: %d"%(i,s))
            ax_list = fig1.axes
            if len(ax_list)==0:
                ax1 = fig1.add_subplot(111,label='ax1')
            else:
                ax1 = ax_list[0]
            
            plotgmmtarget(fig1,ax1,[gmmuk],[gmmf],[],tvec[j],robots,targetset[i].posstates)
            titlt = ax1.get_title()
            titlt = titlt+"  ..  info = %f, .... key = %s"%(dd,str(ykey))
            ax1.set_title(titlt)
            
        else:
            dd = cache[ykey]
            
        # I=I+w[s]*dd
        I.append(dd)
    print("counts for target ",i,"  counts = ",cnts)
    I = np.min(I)
    It = It + I







