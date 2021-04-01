
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
import multiprocessing as mp
import pickle as pkl
from random import shuffle
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
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
from physmodels import transitionmodels as phytm
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
from uq.filters import markovfilter as uqfmarkf
from uq.filters import gmm as uqgmmf
from sklearn import mixture


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
from sensortasking import exhaustive_tasking as stexhaust


# %% plotting functions
markers = ['.','o','s','d','^']

def plotsimIndividTargs(simmanger,targetset,robots,t):
    print("plotsim thread")
    for i in range(targetset.ntargs):
        if targetset[i].isSearchTarget():
            continue
        
        fig = plt.figure("MainSim: Target: %d"%i)
        axlist = fig.axes
        if len(axlist)==0:
            ax = fig.add_subplot(111,label='c')
        else:
            ax = axlist[0]
            ax.cla()
        
        ax.set_title("time step = %f"%(t,) )
        robots[0].mapobj.plotmap(ax)
        xlim=[]
        ylim=[]
        XY=[]
        for r in robots:
            r.plotrobot(ax)
            XY.append(r.mapobj.XYgvec)
        
        XY=np.vstack(XY)
        aamn = np.min(XY,axis=0)
        aamx = np.max(XY,axis=0)
        xlim = [aamn[0]-10,aamx[0]+10]
        ylim = [aamn[1]-10,aamx[1]+10]
        
        
        
        xktruth = targetset[i].groundtruthrecorder.getvar_uptotime_stacked('xtk',t)
        xfku = targetset[i].recorderpost.getvar_uptotime_stacked('xfk',t)
        xfkf = targetset[i].recorderprior.getvar_uptotime_stacked('xfk',t)
        
        if xfku.shape[0]>=xfkf.shape[0]:
            xfk = xfku
        else:
            xfk = xfkf

        ax.plot(xktruth[:,0],xktruth[:,1],linestyle='--',c=targetset[i].color)
        ax.plot(xktruth[-1,0],xktruth[-1,1],c=targetset[i].color,marker='*')
        
        ax.plot(xfk[:,0],xfk[:,1],c=targetset[i].color)
        ax.plot(xfk[-1,0],xfk[-1,1],c=targetset[i].color,marker='s')
        
        ax.annotate(targetset[i].targetName,xktruth[-1,0:2],xktruth[-1,0:2]+2,color=targetset[i].color,fontsize='x-small')
        ax.annotate(targetset[i].targetName,xfk[-1,0:2],xfk[-1,0:2]+2,color=targetset[i].color,fontsize='x-small')

        ax.axis('equal')
        ax.set(xlim=xlim, ylim=ylim)
        plt.show(block=False)
        plt.pause(0.4)
        # k = f'{t:06.2f}'.replace('.','-')
        # simmanger.savefigure(fig, ['SimSnapshot', 'targ',], k+'.png')

def plotsimAllTargs(simmanger,targetset,robots,t,tvec,plotest=True,plotrobottraj=True,plotsearch=True):
    
    xlim=[]
    ylim=[]
    XY=[]
    for r in robots:
        XY.append(r.mapobj.XYgvec)
    
    XY=np.vstack(XY)
    aamn = np.min(XY,axis=0)
    aamx = np.max(XY,axis=0)
    xlim = [aamn[0]-10,aamx[0]+10]
    ylim = [aamn[1]-10,aamx[1]+10]
    
    # plots any searchtargets markov grid first
    if plotsearch:
        fig = plt.figure("Search targs")
        axlist = fig.axes
        if len(axlist)==0:
            ax = fig.add_subplot(projection='3d')
        else:
            ax = axlist[0]
            ax.cla()
        ax.set_title("time step = %f"%(t,) )
        for r in robots:
            r.plotrobotTraj(ax,tvec[:-1])
            r.plotrobot(ax)
        cnt=0
        for i in range(targetset.ntargs):
            if targetset[i].isSearchTarget() is False:
                continue
            xk = targetset[i].dynModel.X
            pk = targetset[i].recorderpost.getvar_bytime('xfk',t)
            # pk = targetset[i].xfk
            ax.scatter(xk[:,0], xk[:,1], pk, marker='.') #markers[cnt]
            cnt+=1
        
        # ax.axis('equal')
        ax.set(xlim=xlim, ylim=ylim)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show(block=False)
        plt.pause(0.1)
    
        
        
        fig = plt.figure("Search MI")
        axlist = fig.axes
        if len(axlist)==0:
            ax = fig.add_subplot(projection='3d')
        else:
            ax = axlist[0]
            ax.cla()
        ax.set_title("time step = %f"%(t,) )
        for r in robots:
            r.plotrobotTraj(ax,tvec[:-1])
            r.plotrobot(ax)
        cnt=0
        for i in range(targetset.ntargs):
            if targetset[i].isSearchTarget() is False:
                continue
            xk = targetset[i].dynModel.X
            pf=targetset[i].recorderprior.getvar_bytime('xfk',t)
            pu=targetset[i].recorderpost.getvar_bytime('xfk',t)
            pf[pf==0]=1e-3
            pu[pu==0]=1e-3
            Hf = -(pf*np.log(pf)+(1-pf)*np.log(1-pf))
            Hu = -(pu*np.log(pu)+(1-pu)*np.log(1-pu))
            MI = Hu
            
            # pk = targetset[i].xfk
            ax.scatter(xk[:,0], xk[:,1], MI, marker='.') #markers[cnt]
            cnt+=1
        
        ax.set(xlim=xlim, ylim=ylim, zlim=[-0.5,1])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show(block=False)
        plt.pause(0.1)
        
        
    # now plot all the rest of the targets
    
    fig = plt.figure("MainSim-all targs")
    axlist = fig.axes
    if len(axlist)==0:
        ax = fig.add_subplot(111,label='contour')
    else:
        ax = axlist[0]
        ax.cla()
        
    ax.set_title("time step = %f"%(t,) )
    
    robots[0].mapobj.plotmap(ax)
    
    for r in robots:
        r.plotrobotTraj(ax,tvec[:-1])
        r.plotrobot(ax)      

    for i in range(targetset.ntargs):
        if targetset[i].isSearchTarget():
            continue
            
        xktruth = targetset[i].groundtruthrecorder.getvar_uptotime_stacked('xtk',t)
        xfku = targetset[i].recorderpost.getvar_uptotime_stacked('xfk',t)
        xfkf = targetset[i].recorderprior.getvar_uptotime_stacked('xfk',t)
        
        Pfku = targetset[i].recorderpost.getvar_bytime('Pfk',t)
        
        ax.plot(xktruth[:,0],xktruth[:,1],linestyle='--',c=targetset[i].color)
        ax.plot(xktruth[-1,0],xktruth[-1,1],c=targetset[i].color,marker='*')
        ax.annotate(targetset[i].targetName,xktruth[-1,0:2],xktruth[-1,0:2]+2,color=targetset[i].color,fontsize='x-small')
        
        
        if plotest:
            if xfku is not None:
                ax.plot(xfku[:,0],xfku[:,1],c=targetset[i].color)
                ax.plot(xfku[-1,0],xfku[-1,1],c=targetset[i].color,marker='s')
                ax.annotate(targetset[i].targetName,xfku[-1,0:2],xfku[-1,0:2]+2,color=targetset[i].color,fontsize='x-small')
    
                XX=utpltshp.getCovEllipsePoints2D(xfku[-1,0:2],Pfku[0:2,0:2],nsig=1,N=100)
                ax.plot(XX[:,0],XX[:,1],color=targetset[i].color)
            
            elif xfkf is not None:             
                ax.plot(xfkf[:,0],xfkf[:,1],c=targetset[i].color)
                ax.plot(xfkf[-1,0],xfkf[-1,1],c=targetset[i].color,marker='s')
                ax.annotate(targetset[i].targetName,xfkf[-1,0:2],xfkf[-1,0:2]+2,color=targetset[i].color,fontsize='x-small')
            else:
                print("Both xfku and xfkf are None for target: #%d"%i)

    ax.axis('equal')
    ax.set(xlim=xlim, ylim=ylim)
    plt.show(block=False)
    plt.pause(0.1)
    k = f'{t:06.2f}'.replace('.','-')
    # simmanger.savefigure(fig, ['SimSnapshot', 'Alltargs'], k+'.png',data=[targetset,robots,t,tvec])



# %% script-level properties

runfilename = __file__
metalog="""
Journal paper on Dynamic sensor tasking paper simulations. Mostly submitted to TAES
The method is to show different sequential exhaustive searches
Author: Venkat
Date: March 27 2021

"""


simmanger = uqsimmanager.SimManager(t0=0,tf=200,dt=2,dtplot=0.1,
                                  simname="DynamicSensorTasking-SeqExhaust",savepath="simulations",
                                  workdir=os.getcwd())

simmanger.initialize()


# %% Create Map

rgmap = gridmap.Regular2DNodeGrid(xy0=(0,0),xyf=(101,101),d=(10,10))

rgmap.th=np.array([0])
fig = plt.figure()
ax = fig.add_subplot(111)
rgmap.plotmap(ax)
plt.pause(0.1)



# %% Set UAVs


robot = robtgr.Robot2DRegGrid()
maxvmag = 5 #5
minvmag = 1 #5
maxturnrate = 2.5 #3
robot.NominalVmag = simmanger.data['robot.NominalVmag'] = 2 #2
robot.MinTempXferr = simmanger.data['robot.MinTempXferr'] = 0.5

robot.dynModel = phymm.KinematicModel_CT_control(L1=0.016, L2=0.0001,maxvmag=maxvmag,minvmag=minvmag,maxturnrate=maxturnrate)
robot.mapobj = rgmap

robot.xk=np.array([6,6,0])
robot.xk =  rgmap.snap2grid(robot.xk)

mapobj = robot.mapobj
    
with utltm.TimingContext():
    robttj.generateTemplates_reachSet(robot,min([0.5,simmanger.dt/20]),simmanger.dt)

# for uk_key in robot.iterateControlsKeys(robot.xk[0:2],robot.xk[2]):
#     print(uk_key)
#     robot.plotdebugTemplate(uk_key)
#     plt.pause(2)
    


# now setting the robots
robots=[robtgr.Robot2DRegGrid(),robtgr.Robot2DRegGrid()]




robots[0].robotName= 'UAV:0'
robots[0].dynModel = phymm.KinematicModel_CT_control(L1=0.016, L2=0.0001,maxvmag=maxvmag,minvmag=minvmag,maxturnrate=maxturnrate)
robots[0].mapobj = robot.mapobj
robots[0].controltemplates = copy.deepcopy(robot.controltemplates)
robots[0].xk=rgmap.snap2grid(np.array([70,70,0]))
robots[0].robotColor = 'b'
robots[0].sensormodel=physmfov.XYcircularFOVsensor(R=block_diag((0.1)**2, (0.1)**2), 
                                                    FOVradius=20,
                                                    posstates=[0,1],
                                                    FOVcolor='b',
                                                    TP=0.9, TN=0.9, FP=0.1, FN=0.1,
                                                    recordSensorState=True,
                                                    enforceConstraint=True)
robots[0].updateSensorModel()


robots[1].robotName= 'UAV:1'
robots[1].dynModel = phymm.KinematicModel_CT_control(L1=0.016, L2=0.0001,maxvmag=maxvmag,minvmag=minvmag,maxturnrate=maxturnrate)
robots[1].mapobj = robot.mapobj
robots[1].controltemplates = copy.deepcopy(robot.controltemplates)
robots[1].xk=rgmap.snap2grid(np.array([30,30,0]))
robots[1].robotColor = 'r'
robots[1].sensormodel=physmfov.XYcircularFOVsensor(R=block_diag((0.1)**2, (0.1)**2), 
                                                   posstates=[0,1], 
                                                   FOVradius=20,
                                                    FOVcolor='r',
                                                    TP=0.9, TN=0.9, FP=0.1, FN=0.1,
                                                    recordSensorState=True,
                                                    enforceConstraint=True)
robots[1].updateSensorModel()   



# robots.pop(1)



# %% Create Targets
NUM_COLORS = 200

colors = [colmap(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
shuffle(colors)

targetset = phytarg.TargetSet()
vmag = 0.3
DD = []
flname = "testcases/target.pkl"
# if os.path.isfile(flname):
#     with open(flname,'rb') as F:
#         DD=pkl.load(F)

xlim=[]
ylim=[]
XY=[]
for r in robots:
    r.plotrobot(ax)
    XY.append(r.mapobj.XYgvec)

XY=np.vstack(XY)
aamn = np.min(XY,axis=0)
aamx = np.max(XY,axis=0)
xlim = [aamn[0],aamx[0]]
ylim = [aamn[1],aamx[1]]

for i in range(2):

    while(1):
        xf0 = np.random.rand(4)
        
        xf0[0] = (xlim[1]-xlim[0])*xf0[0]+xlim[0]
        xf0[1] = (ylim[1]-ylim[0])*xf0[1]+ylim[0]
        
        xf0[2:4] = np.random.randn(2)
        xf0[2:4] = vmag*xf0[2:4]/nplg.norm(xf0[2:4])
        # xf0[2:4] = vmag*xf0[2:4]
        # xf0[4] = 0.01*np.random.randn()
        # Pf0 = np.random.randn(4,4)
        # Pf0 = np.matmul(Pf0,Pf0.T)
        # Pf0[0:2,0:2]=10*Pf0[0:2,0:2]/np.max(Pf0[0:2,0:2].reshape(-1))
        # Pf0[2:4,2:4]=0.01*Pf0[2:4,2:4]/np.max(Pf0[2:4,2:4].reshape(-1))
        Pf0 = block_diag(5**2,5**2,0.2**2,0.2**2)
        u,v=nplg.eig(Pf0)
        if np.all(u>0):
            break
    
    # Pf0[4,4]=0.0001
    
    # if i==0:
    #     Pf0[0:2,0:2]=15*Pf0[0:2,0:2]
    # DD.append({'xf0':xf0,'Pf0':Pf0})
        

    
    dynmodel = phymm.KinematicModel_UM_control(L1=0.0016, L2=1e-8,maxturnrate=0.25,maxvmag=vmag)

    recorderobjprior = uqrecorder.StatesRecorder_list(statetypes = ['xfk','Pfk'])
    recorderobjpost = uqrecorder.StatesRecorder_list(statetypes = ['xfk','Pfk'])
    target = phytarg.Target(dynModel=dynmodel, xfk=xf0, Pfk=Pf0, currt = 0, recordfilterstate=True,
                 recorderobjprior = recorderobjprior,recorderobjpost=recorderobjpost)
    target.groundtruthrecorder = uqrecorder.StatesRecorder_list(statetypes = ['xtk','uk'] )
    target.setInitialdata(0,xfk=xf0, Pfk=Pf0)
    target.freeze(recorderobj=False)
    target.groundtruthrecorder.record(0, xtk=xf0,uk=None)
    
    target.targetName = "Trgt:%d"%i
    target.color=colors[i]
    targetset.addTarget(target)


xg = np.arange(rgmap.xy0[0],rgmap.xyf[0],10)
yg = np.arange(rgmap.xy0[1],rgmap.xyf[1],10)

Xgmesh,Ygmesh=np.meshgrid(xg, yg )
XYgvec = np.hstack([ Xgmesh.reshape(-1,1),Ygmesh.reshape(-1,1) ])
P=np.array([[0.99,0.01],[0.01,0.99]])
dynmodel = phytm.TransitionModel(P,XYgvec)
xf0 = 0.5*np.ones(dynmodel.fn)
recorderobjprior = uqrecorder.StatesRecorder_list(statetypes = ['xfk'])
recorderobjpost = uqrecorder.StatesRecorder_list(statetypes = ['xfk'])
target = phytarg.Target(dynModel=dynmodel, xfk=xf0, Pfk=None, currt = 0, recordfilterstate=True,
             recorderobjprior = recorderobjprior,recorderobjpost=recorderobjpost)
target.targtype = phytarg.TargetType.Search
target.setInitialdata(0,xfk=xf0, Pfk=None)
target.filterer = uqfmarkf.TargetMarkovF()
target.freeze(recorderobj=False)

target.targetName = "SearchTarget"
target.color=colors[-1]
targetset.addTarget(target)   


if not os.path.isfile(flname):
    with open(flname,'wb') as F:
        pkl.dump(DD,F)


# %% Generate truth
# fig = plt.figure("Search targs")
# ax = fig.add_subplot(projection='3d')
# i=2
# xk = targetset[i].dynModel.X
# pk = targetset[i].xfk
# ax.scatter(xk[:,0], xk[:,1], pk, marker='.')

simmanger.data['changeRate'] = changeRate = 0.99

plt.close("all")
maxturnrate_diff = 0.1

for t,tk,dt in simmanger.iteratetimesteps():
    print (t,tk,dt)
    
    Uk = []
    # generate truth at time t+dt
    for i in range(targetset.ntargs):
        if targetset[i].isSearchTarget():
            continue
        
        xk = targetset[i].groundtruthrecorder.getvar_bytime('xtk',t)
        vv = xk[2:4]
        vmag = nplg.norm(xk[2:4])
        if vmag >0:
            vdir = xk[2:4]/vmag
        else:
            vdir = np.array([0,0])
        
        # maxturnrate = targetset[i].dynModel.maxturnrate
        maxvmag = targetset[i].dynModel.maxvmag
        
        if not robots[0].mapobj.isInMap(xk):
            dirnmid = robots[0].mapobj.middirn()-xk[0:2]
            dirnmid = dirnmid/nplg.norm(dirnmid)
            vdir = 0.5*(vdir+dirnmid)
            vdir = vdir/nplg.norm(vdir)
            vv = vmag*vdir
            # c=np.cross(np.hstack([vdir,0]),np.hstack([dirnmid,0]))
            # turnrate = np.sign(c[2])*np.arccos(np.dot(vdir,dirnmid))/(simmanger.dt*5)
            uk = vv
        else:
            # random turn of target
            ss  = np.random.rand()
            if ss>changeRate:
                d = np.random.randn(2)
                # turnrate = xk[4]+max([min([d,maxturnrate_diff]),-maxturnrate_diff])
                vdir = 0.7*vdir+0.3*d
                vdir = vdir/nplg.norm(vdir)
                vv = vmag*vdir
                uk = vv
            else:
                uk = None
                
        Uk.append(uk)
        # print("uk = ",uk)
        # uk=None
        _,xk1=targetset[i].dynModel.propforward( t, dt, xk, uk=uk)
        targetset[i].groundtruthrecorder.recordupdate(t+dt,xtk=xk1,uk=uk)
    


 
    # Plotting of simulations
    # plotsimIndividTargs(simmanger,targetset,robots,t+dt)
    tvec = simmanger.tvec[simmanger.tvec<=(tk+dt)]
    plotsimAllTargs(simmanger,targetset,robots,t+dt,tvec,plotest=False,plotrobottraj=False,plotsearch=False)
    
    plt.ion()
    plt.show()
    plt.pause(0.01)
    
simtestcase = "testcases/simcase.pkl"
with open(simtestcase,'wb') as F:
    pkl.dump({'targetset':targetset,'robots':robots,'simmanger':simmanger},F)
# %% Run sim
with open(simtestcase,'rb') as F:
    data = pkl.load(F)
    targetset=data['targetset']
    robots=data['robots']
    simmanger=data['simmanger']

searchMIwt = simmanger.data['searchMIwt'] = 3



# Targetfilterer = uqfsigf.TargetSigmaKF( sigmamethod=quadcub.SigmaMethod.UT )
Targetfilterer = uqkf.TargetEKF()
InfoTargetfilterer = uqkf.TargetEKF()

mapobj = robot.mapobj


simmanger.data['TT'] = TT = 5
simmanger.data['TTrecomp'] = TTrecomp = 3
 
dptvec = [0]

for t,tk,dt in simmanger.iteratetimesteps():
    print (t,tk,dt)
    
    
    if t==0 or t==dptvec[min([TTrecomp,len(dptvec)-1])]:
        dptvec = simmanger.tvec[tk:min([tk+TT,simmanger.ntimesteps]) ]
        stexhaust.exhaustive_seq_robot(dptvec,robots,targetset,Targetfilterer,searchMIwt=searchMIwt)
        # stexhaust.random_seq_robot(dptvec,robots,targetset,Targetfilterer,searchMIwt=0.5)
        


    
        
        
    
    Ukfilter = [None]*targetset.ntargs
    # propagate the targets
    for i in range(targetset.ntargs):
        if targetset[i].isSearchTarget():
            targetset[i].filterer.propagate( t,dt,targetset[i],updttarget=True)
        else:
            Targetfilterer.propagate(t,dt,targetset[i],Ukfilter[i],updttarget=True)
    
    #### TIME IS NOW AT t+dt
    
    
    # update the robots to t+dt
    for j in range(len(robots)):
        # uk = robots[j].getcontrol(t)
        robots[j].xk = robots[j].statehistory[t+dt]
        robots[j].updateTime(t+dt)
        robots[j].updateSensorModel()
        
    # do measurment update for targets
    with utltm.TimingContext("Measureemt update: "):
        for i in range(targetset.ntargs):
            if targetset[i].isSearchTarget():
                for r in range(len(robots)):
                    sensormodel = robots[r].sensormodel
                    xk = targetset[i].dynModel.X
                    zk,isinsidek,Lk = sensormodel.generateRndDetections( t, dt, xk,useRecord=False)
                    targetset[i].filterer.measUpdate( t+dt, dt, targetset[i], sensormodel,zk, updttarget=True)
            else:
                for r in range(len(robots)):
                    xk = targetset[i].groundtruthrecorder.getvar_bytime('xtk',t+dt)
                    sensormodel = robots[r].sensormodel
                    zk,isinsidek,Lk = sensormodel.generateRndMeas(t+dt, dt, xk,useRecord=False)
                    if isinsidek == 0:
                        fovSigPtFrac = 1  # dont do measurement update but just copy prior
                        Targetfilterer.measUpdate(t+dt, dt, targetset[i], sensormodel, zk, updttarget=True, fovSigPtFrac=fovSigPtFrac,justcopyprior=True)
                    else:
                        fovSigPtFrac = -1 # do measurement update
                        Targetfilterer.measUpdate(t+dt, dt, targetset[i], sensormodel, zk, updttarget=True, fovSigPtFrac=fovSigPtFrac,justcopyprior=False)
                    
                    
            
    # Plotting of simulations
    # plotsimIndividTargs(simmanger,targetset,robots,t+dt)
    plotsimAllTargs(simmanger,targetset,robots,t+dt,simmanger.tvec[:(tk+dt)])
    
    plt.ion()
    plt.show()
    plt.pause(0.1)
    
    # Designate a target as lost
    FOVradius=[]
    for r in range(len(robots)):
        FOVradius.append( robots[r].sensormodel.FOVradius )
    for i in range(targetset.ntargs):
        if targetset[i].isSearchTarget() is False:
            egval, = nplg.eig(targetset[i].Pfk[0:2,0:2])
            if np.any(np.sqrt(egval))>=max(FOVradius):
                targetset[i].makeInactive()


# %% Finalize and save

simmanger.finalize()

simmanger.save(metalog, mainfile=runfilename, targetset=targetset, robots=robots)



