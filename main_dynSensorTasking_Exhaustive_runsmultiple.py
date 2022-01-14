
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

import matplotlib.pyplot as plt
colmap = plt.get_cmap('gist_rainbow')
# plt.style.use("utils/plotting/production_pyplot_style.txt")


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
from utils import simmanager
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
import pyperclip

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
from sensortasking import exhaustive_tasking_seq_robot as stexhaustseqrobot
from sensortasking import exhaustive_tasking_seq_time as stexhaustseqtime


from utils.metrics import tracking as utmttrack


plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
figsize=(12,9)
# %% plotting functions
markers = ['.','o','s','d','^']

def getActiveTrackTimes(tt,statuses):
    TT=[]
    T=[]
    flg = 0
    prevflg = 0
    for i in range(len(statuses)):
        prevflg = flg
        if statuses[i] == 'Active':
            flg = 1
        else:
            flg = 0
        
        if i == len(statuses)-1 and statuses[i] == 'Active':
            flg=0
            prevflg=1
            
        if prevflg==0 and flg==1:
            T.append(i)
        elif prevflg==1 and flg==0:
            T.append(i)
            TT.append(T)
            T=[]
    return TT
        
def plotsimAllTargs(simngr,pather,groundtargetset,targetset,robots,t,tvec,plotest=True,plotrobottraj=True,plotsearch=True,saveit=False):
    
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
        fig = plt.figure("Search targs",figsize=figsize)
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
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show(block=False)
        plt.pause(0.1)
    
        
        
        fig = plt.figure("Search MI",figsize=figsize)
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
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show(block=False)
        plt.pause(0.1)
        
        
    # now plot all the rest of the targets
    
    fig = plt.figure("MainSim-all targs",figsize=figsize)
    axlist = fig.axes
    if len(axlist)==0:
        ax = fig.add_subplot(111,label='contour')
    else:
        ax = axlist[0]
        ax.cla()
        
    ax.set_title("time step = %f"%(t,) )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    robots[0].mapobj.plotmap(ax)
    
    for r in robots:
        # r.plotrobotTraj(ax,tvec[:-1])
        r.plotrobot(ax)      

    for i in range(groundtargetset.ntargs):
        if groundtargetset[i].isSearchTarget():
            continue
            
        tt,xktruth = groundtargetset[i].groundtruthrecorder.getvar_uptotime_stacked('xtk',t,returntimes=True)
        ax.plot(xktruth[:,0],xktruth[:,1],linestyle='--',c=groundtargetset[i].color)
        ax.plot(xktruth[-1,0],xktruth[-1,1],c=groundtargetset[i].color,marker='*')
        ax.annotate(groundtargetset[i].targetName,xktruth[-1,0:2],xktruth[-1,0:2]-2,color=groundtargetset[i].color)
        
    for i in range(targetset.ntargs):
        if targetset[i].isSearchTarget():
            continue
            
        
        tt,xfku = targetset[i].recorderpost.getvar_uptotime_stacked('xfk',t,returntimes=True)
        tt,status_ku = targetset[i].recorderpost.getvar_uptotime_list('status',t,returntimes=True)
        if status_ku is None or len(status_ku)<2:
            continue
        TT = getActiveTrackTimes(tt,status_ku)
        
        # tt,xfkf = targetset[i].recorderprior.getvar_uptotime_stacked('xfk',t)
        xfkf = None
        
        Pfku = targetset[i].recorderpost.getvar_bytime('Pfk',t)
        
        
        
        

        if plotest:
            if xfku is not None:
                for T in TT:
                    ax.plot(xfku[T[0]:T[1],0],xfku[T[0]:T[1],1],c=targetset[i].color,linewidth=2)
                    # ax.plot(xfku[T[0],0],xfku[T[0],1],c=targetset[i].color,marker='s')
                    # ax.plot(xfku[T[1],0],xfku[T[1],1],c=targetset[i].color,marker='o')
                    
                ax.plot(xfku[-1,0],xfku[-1,1],c=targetset[i].color,marker='s')
                ax.annotate(targetset[i].targetName,xfku[-1,0:2],xfku[-1,0:2]+2,color=targetset[i].color)
    
                XX=utpltshp.getCovEllipsePoints2D(xfku[-1,0:2],Pfku[0:2,0:2],nsig=1,N=100)
                ax.plot(XX[:,0],XX[:,1],color=targetset[i].color)
            
            elif xfkf is not None:             
                ax.plot(xfkf[:,0],xfkf[:,1],c=targetset[i].color,linewidth=2)
                ax.plot(xfkf[-1,0],xfkf[-1,1],c=targetset[i].color,marker='s')
                ax.annotate(targetset[i].targetName,xfkf[-1,0:2],xfkf[-1,0:2]+2,color=targetset[i].color)
            else:
                print("Both xfku and xfkf are None for target: #%d"%i)

    ax.axis('equal')
    ax.set(xlim=xlim, ylim=ylim)
    plt.show(block=False)
    plt.pause(0.1)
    fname = f'{t:06.0f}'.replace('.','-')
    if saveit:
        simngr.savefigure(fig, pather, fname,figformat='.png',data=[simngr,groundtargetset,targetset,robots,t,tvec])



# %% script-level properties

runfilename = __file__
metalog="""
Journal paper on Dynamic sensor tasking paper simulations. Mostly submitted to TAES
The method is to show different sequential exhaustive searches
Author: Venkat
Date: March 27 2021

"""


simngr = simmanager.SimManager(t0=0,tf=500,dt=2,dtplot=0.1,
                                  simname="DynamicSensorTasking-SeqExhaust-Time",savepath="simulations",
                                  workdir=os.getcwd())

simngr.initialize()


# %% Create Map

rgmap = gridmap.Regular2DNodeGrid(xy0=(0,0),xyf=(101,101),d=(10,10))

rgmap.th=np.array([0])
fig = plt.figure()
ax = fig.add_subplot(111)
rgmap.plotmap(ax,labelnodes=True)
plt.pause(0.1)



# %% Set UAVs


robot = robtgr.Robot2DRegGrid()
maxvmag = 5 #5
minvmag = 1 #5
maxturnrate = 2.5 #3
robot.NominalVmag = simngr.data['robot.NominalVmag'] = 2 #2
robot.MinTempXferr = simngr.data['robot.MinTempXferr'] = 0.5

robot.dynModel = phymm.KinematicModel_CT_control(L1=0.016, L2=0.0001,maxvmag=maxvmag,minvmag=minvmag,maxturnrate=maxturnrate)
robot.mapobj = rgmap

robot.xk=np.array([6,6,0])
robot.xk =  rgmap.snap2grid(robot.xk)

mapobj = robot.mapobj
    
with utltm.TimingContext():
    robttj.generateTemplates_reachSet(robot,min([0.5,simngr.dt/20]),simngr.dt)

# for uk_key in robot.iterateControlsKeys(robot.xk[0:2],robot.xk[2]):
#     print(uk_key)
#     robot.plotdebugTemplate(uk_key)
#     plt.pause(2)
    


# now setting the robots
robots=[robtgr.Robot2DRegGrid(),robtgr.Robot2DRegGrid(),robtgr.Robot2DRegGrid()]




robots[0].robotName= 'UAV:0'
robots[0].dynModel = phymm.KinematicModel_CT_control(L1=0.016, L2=0.0001,maxvmag=maxvmag,minvmag=minvmag,maxturnrate=maxturnrate)
robots[0].mapobj = robot.mapobj
robots[0].controltemplates = copy.deepcopy(robot.controltemplates)
robots[0].xk=rgmap.snap2grid(np.array([70,70,0]))
robots[0].robotColor = 'b'
robots[0].shape = {'a':2,'w':1}
robots[0].sensormodel=physmfov.XYcircularFOVsensor(R=block_diag((1)**2, (1)**2), 
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
robots[1].shape = {'a':2,'w':1}
robots[1].sensormodel=physmfov.XYcircularFOVsensor(R=block_diag((0.2)**2, (0.2)**2), 
                                                   posstates=[0,1], 
                                                   FOVradius=10,
                                                    FOVcolor='r',
                                                    TP=0.9, TN=0.9, FP=0.1, FN=0.1,
                                                    recordSensorState=True,
                                                    enforceConstraint=True)
robots[1].updateSensorModel()   


robots[2].robotName= 'UAV:2'
robots[2].dynModel = phymm.KinematicModel_CT_control(L1=0.016, L2=0.0001,maxvmag=maxvmag,minvmag=minvmag,maxturnrate=maxturnrate)
robots[2].mapobj = robot.mapobj
robots[2].controltemplates = copy.deepcopy(robot.controltemplates)
robots[2].xk=rgmap.snap2grid(np.array([70,30,0]))
robots[2].robotColor = 'r'
robots[2].shape = {'a':2,'w':1}
robots[2].sensormodel=physmfov.XYcircularFOVsensor(R=block_diag((0.2)**2, (0.2)**2), 
                                                   posstates=[0,1], 
                                                   FOVradius=10,
                                                    FOVcolor='r',
                                                    TP=0.9, TN=0.9, FP=0.1, FN=0.1,
                                                    recordSensorState=True,
                                                    enforceConstraint=True)
robots[2].updateSensorModel()   

# robots.pop(1)



# %% Create Targets
NUM_COLORS = 200

colors = [colmap(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
shuffle(colors)
for casecnt in range(20):
    groundtargetset = phytarg.TargetSet()
    targetset = phytarg.TargetSet()
    vmag = 0.3
    DD = []
    # flname = "testcases/target.pkl"
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
    
    for i in range(10):
    
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
    
        recorderobjprior = uqrecorder.StatesRecorder_list(statetypes = ['xfk','Pfk','status'])
        recorderobjpost = uqrecorder.StatesRecorder_list(statetypes = ['xfk','Pfk','status'])
        target = phytarg.Target(dynModel=dynmodel, xfk=xf0, Pfk=Pf0, currt = 0, recordfilterstate=True,
                     recorderobjprior = recorderobjprior,recorderobjpost=recorderobjpost)
        target.groundtruthrecorder = uqrecorder.StatesRecorder_list(statetypes = ['xtk','uk'] )
        target.setInitialdata(0,xfk=xf0, Pfk=Pf0)
        target.freeze(recorderobj=False)
        target.groundtruthrecorder.record(0, xtk=xf0,uk=None)
        
        target.targetName = "Trgt:%d"%i
        target.color=colors[i]
        groundtargetset.addTarget(target)
    
    
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



    simngr.data['changeRate'] = changeRate = 0.7
    
    plt.close("all")
    maxturnrate_diff = 0.1

    for t,tk,dt in simngr.iteratetimesteps():
        print (t,tk,dt)
        
        Uk = []
        # generate truth at time t+dt
        for i in range(groundtargetset.ntargs):
            if groundtargetset[i].isSearchTarget():
                continue
            
            xk = groundtargetset[i].groundtruthrecorder.getvar_bytime('xtk',t)
            vv = xk[2:4]
            vmag = nplg.norm(xk[2:4])
            if vmag >0:
                vdir = xk[2:4]/vmag
            else:
                vdir = np.array([0,0])
            
            # maxturnrate = targetset[i].dynModel.maxturnrate
            maxvmag = groundtargetset[i].dynModel.maxvmag
            
            if not robots[0].mapobj.isInMap(xk):
                dirnmid = robots[0].mapobj.middirn()-xk[0:2]
                dirnmid = dirnmid/nplg.norm(dirnmid)
                vdir = 0.5*(vdir+dirnmid)
                vdir = vdir/nplg.norm(vdir)
                vv = vmag*vdir
                # c=np.cross(np.hstack([vdir,0]),np.hstack([dirnmid,0]))
                # turnrate = np.sign(c[2])*np.arccos(np.dot(vdir,dirnmid))/(simngr.dt*5)
                uk = vv
            else:
                # random turn of target
                ss  = np.random.rand()
                if ss>changeRate:
                    d = np.random.randn(2)
                    # turnrate = xk[4]+max([min([d,maxturnrate_diff]),-maxturnrate_diff])
                    vdir = 0.9*vdir+0.1*d
                    vdir = vdir/nplg.norm(vdir)
                    vv = vmag*vdir
                    uk = vv
                else:
                    uk = None
                    
            Uk.append(uk)
            # print("uk = ",uk)
            # uk=None
            _,xk1=groundtargetset[i].dynModel.propforward( t, dt, xk, uk=uk)
            groundtargetset[i].groundtruthrecorder.record(t+dt,xtk=xk1,uk=uk)
        
    
    
     
        # Plotting of simulations
        # plotsimIndividTargs(simngr,targetset,robots,t+dt)
        tvec = simngr.tvec[simngr.tvec<=(tk+dt)]
        # plotsimAllTargs(simngr,None,groundtargetset,targetset,robots,t+dt,tvec,plotest=False,plotrobottraj=False,plotsearch=False)
        
        # plt.ion()
        # plt.show()
        # plt.pause(0.01)
    
    # plotsimAllTargs(simngr,None,groundtargetset,targetset,robots,t+dt,tvec,plotest=False,plotrobottraj=False,plotsearch=False)
    simtestcase = simngr.createFile("testcase_%03d.pkl"%casecnt,pather=[],addtimetag=False)
    with open(simtestcase,'wb') as F:
        pkl.dump({'groundtargetset':groundtargetset,'targetset':targetset,'robots':robots},F)
    
    
    
def RE_Initialize(zk,target,dt):
    Pf0 = block_diag(5**2,5**2,0.2**2,0.2**2)
    xf0 = np.hstack([zk,0.1,0.1])
    
        
    # estimate rough xf0 after two measurements
    if 'ZK' in target.tempData:
        target.tempData['ZK'].append((t+dt,zk))
        ddt=target.tempData['ZK'][1][0]-target.tempData['ZK'][0][0]
        vf=(target.tempData['ZK'][1][1]-target.tempData['ZK'][0][1])/ddt
        if ddt<=2*dt:
            xf0=np.hstack([zk,vf])
        
        target.tempData.pop('ZK')
        
    else:
        target.tempData['ZK']=[(t+dt,zk)]
    
    return xf0,Pf0
# %% Run sim
# loadtestcase = simtestcase
# loadtestcase = "simulations/DynamicSensorTasking-SIMset1/DynamicSensorTasking-SeqExhaust-Robot_0_2021-04-09-14H-44M-44s/testcase.pkl"
# with open(loadtestcase,'rb') as F:
#     data = pkl.load(F)
#     robots=data['robots']
savefolder=os.path.join('simulations','DynamicSensorTasking-SeqExhaust-Time_0_2021-10-16-11H-44M-51s')


# methods=['MethodType',Ns,lambda] where Ns is the number of robots
methods=[('Seq-UAV',2,3),('Seq-UAV',2,10),('Seq-UAV',2,20),('Seq-Time',2,3),('Seq-Time',2,10),('Seq-Time',2,20),('Seq-UAV',3,3),('Seq-UAV',3,10),('Seq-UAV',3,20)]
for mth in methods:
    for casecnt in range(10):
        loadtestcase = os.path.join(savefolder,"testcase_%03d.pkl"%casecnt)
        with open(loadtestcase,'rb') as F:
            data = pkl.load(F)
            groundtargetset = data['groundtargetset']
            targetset=data['targetset']
            robots=data['robots']
            # simngr=data['simngr']
        
        robots = robots[:mth[1]]
        searchMIwt = simngr.data['searchMIwt'] = mth[2]
        
        
        
        # Targetfilterer = uqfsigf.TargetSigmaKF( sigmamethod=quadcub.SigmaMethod.UT )
        Targetfilterer = uqkf.TargetEKF()
        InfoTargetfilterer = uqkf.TargetEKF()
        
        mapobj = robot.mapobj
        
        
        simngr.data['TT'] = TT = 5
        simngr.data['TTrecomp'] = TTrecomp = 3
        
        simngr.data['LOST_FOV_FACTOR'] = LOST_FOV_FACTOR = 1
         
        dptvec = [0]
        
        for t,tk,dt in simngr.iteratetimesteps():
            print (t,tk,dt)
            
            # -------------Do detection, initialize new targets, re-initialize previously lost targets----------------------
            Tt = t
            for i in range(groundtargetset.ntargs): 
                gtargID = groundtargetset[i].ID
                # if target is already tracked
                if targetset.hasTarget(gtargID):
                    target = targetset.getTargetByID(gtargID)
                    # check if it is active or inactive
                    if target.isActive():
                        pass # do nothing
                    else: # is InActive: check if any robot sees it and re-initialize
                        for r in range(len(robots)):
                            sensormodel = robots[r].sensormodel
                            xtk = groundtargetset[i].groundtruthrecorder.getvar_bytime('xtk',Tt)
                            zk,isinsidek,Lk = sensormodel.generateRndMeas(Tt, dt, xtk,useRecord=False)
                            if isinsidek == 0: # outside FOV
                                pass # cannot re-initialize a robot outside 
                            else:
                               
                                if sensormodel.sensStateStrs!=['x','y']:
                                    raise NotImplemented("You need to somehow get x and y from the sensor to RE-initialize target")
                                
                                target.makeActive()
                                print("Target %s is now Active"% target.targetName)
                                xf0,Pf0 = RE_Initialize(zk,target,dt)
                                
                                target.setInitialdata(Tt,xfk=xf0, Pfk=Pf0)
                                break
                            
                else:# if target is not tracked at all
                    
                    for r in range(len(robots)):
                        sensormodel = robots[r].sensormodel
                        xtk = groundtargetset[i].groundtruthrecorder.getvar_bytime('xtk',Tt)
                        zk,isinsidek,Lk = sensormodel.generateRndMeas(Tt, dt, xtk,useRecord=False)
                        if isinsidek == 0: # outside FOV
                            pass # cannot initialize a robot outside 
                        else:
                            if sensormodel.sensStateStrs!=['x','y']:
                                raise NotImplemented("You need to somehow get x and y from the sensor to initialize target")
                                
                            target =  groundtargetset[i].makeCopy()
                            target.makeActive()
                            
                            target.groundtruthrecorder = None
                            target.recorderprior.cleardata(keepInitial = False)
                            target.recorderpost.cleardata(keepInitial = False)
                            
                            
                            print("Target %s is now Tracked "% target.targetName)
                            
                            xf0,Pf0 = RE_Initialize(zk,target,dt)
                            
                            target.setInitialdata(Tt,xfk=xf0, Pfk=Pf0)
                            
                            targetset.addTarget(target)
                            break
            # ------------------------------------------------------------------------
            #                           ROBOT TRAJECTORY
            #-------------------------------------------------------------------------
            if t==0 or t==dptvec[min([TTrecomp,len(dptvec)-1])]:
                dptvec = simngr.tvec[tk:min([tk+TT,simngr.ntimesteps]) ]
                if mth[0]=='Seq-UAV':
                    stexhaustseqrobot.exhaustive_seq_robot(dptvec,robots,targetset,Targetfilterer,searchMIwt=searchMIwt)
                else:
                    stexhaustseqtime.exhaustive_seq_time(dptvec,robots,targetset,Targetfilterer,searchMIwt=searchMIwt)
                
        
            # -----------------------------------------------------------------
            
                
                
            
            Ukfilter = [None]*targetset.ntargs
            # propagate the targets
            for i in range(targetset.ntargs):
                if targetset[i].isSearchTarget():
                    targetset[i].filterer.propagate( t,dt,targetset[i],updttarget=True)
                else:
                    if targetset[i].isInActive(): 
                        Targetfilterer.propagate(t, dt, targetset[i],Ukfilter[i],updttarget=True, justcopyprior=True)
                        continue
                    Targetfilterer.propagate(t,dt,targetset[i],Ukfilter[i],updttarget=True, justcopyprior=False)
            
            #### TIME IS NOW AT t+dt
            
            
            # update the robots to t+dt
            for j in range(len(robots)):
                # uk = robots[j].getcontrol(t)
                robots[j].xk = robots[j].statehistory[t+dt]
                robots[j].updateTime(t+dt)
                robots[j].updateSensorModel()
            
            
        
                   
            # do measurment update for targets that are being tracked (Active). Does not matter if in FOV or not,deal with those cases as is
            # with utltm.TimingContext("Measureemt update: "):
            for i in range(targetset.ntargs):
                if targetset[i].isSearchTarget():
                    for r in range(len(robots)):
                        sensormodel = robots[r].sensormodel
                        xk = targetset[i].dynModel.X
                        zk,isinsidek,Lk = sensormodel.generateRndDetections( t+dt, dt, xk,useRecord=False)
                        targetset[i].filterer.measUpdate( t+dt, dt, targetset[i], sensormodel,zk, updttarget=True)
                else:
                    if targetset[i].isInActive(): 
                        Targetfilterer.measUpdate(t+dt, dt, targetset[i], sensormodel, zk, updttarget=True, justcopyprior=True)
                        continue
                    
                    for r in range(len(robots)):
                        targID = targetset[i].ID
                        
                        xtk = groundtargetset.getTargetByID(targID).groundtruthrecorder.getvar_bytime('xtk',t+dt)
                        sensormodel = robots[r].sensormodel
                        zk,isinsidek,Lk = sensormodel.generateRndMeas(t+dt, dt, xtk,useRecord=False)
                        if isinsidek == 0: # outside FOV
                            fovSigPtFrac = 1  # dont do measurement update but just copy prior
                            Targetfilterer.measUpdate(t+dt, dt, targetset[i], sensormodel, zk, updttarget=True, fovSigPtFrac=fovSigPtFrac,justcopyprior=True)
                        else: # inside FOV
                            fovSigPtFrac = -1 # do measurement update
                            Targetfilterer.measUpdate(t+dt, dt, targetset[i], sensormodel, zk, updttarget=True, fovSigPtFrac=fovSigPtFrac,justcopyprior=False)
                            
                            
                    
            # Plotting of simulations
            # plotsimIndividTargs(simngr,targetset,robots,t+dt)
            # plotsimAllTargs(simngr,['SimSnapshot'],groundtargetset,targetset,robots,t+dt,simngr.tvec[:(tk+dt)],saveit=True)
            
            # plt.ion()
            # plt.show()
            # plt.pause(0.1)
            
            # Designate a target as lost
            FOVradius=[]
            for r in range(len(robots)):
                FOVradius.append( robots[r].sensormodel.FOVradius )
            for i in range(targetset.ntargs):
                if targetset[i].isSearchTarget() is False:
                    egval,egvec = nplg.eig(targetset[i].Pfk[0:2,0:2])
                    if np.any(np.sqrt(egval) >= LOST_FOV_FACTOR*max(FOVradius)):
                        targetset[i].makeInactive()
                        print("Target %s is now InActive"% targetset[i].targetName)
        
                       
        metrics,DF,figs = utmttrack.multiTarget_tracking_metrics(t,groundtargetset,targetset,plotfigs=False)

        simtestcase = simngr.createFile("runcase_%s_Ns%02d_L%02d_%03d.pkl"%(mth[0],mth[1],mth[2],casecnt),pather=['runs'],addtimetag=False)
        with open(simtestcase,'wb') as F:
            pkl.dump({'groundtargetset':groundtargetset,'targetset':targetset,'robots':robots,'metrics':metrics,'DF':DF},F)



# alphabets = list('ABCDEFGHIJKLMNOPQRTSUVWXYZ')
# cols = metrics.columns
# cols_dict = {cols[i]:alphabets[i] for i in range(len(cols))}
# ss = metrics.rename(columns=cols_dict).to_latex(index=True)
# print(ss)
# pyperclip.copy(ss)


# for ff in figs:
#     fname = ff.get_label()
#     simngr.savefigure(ff, ['Metrics'], fname,figformat='.png')

# %% Finalize and save

simngr.finalize()

simngr.save(metalog, mainfile=runfilename,metrics=metrics, targetset=targetset,groundtargetset=groundtargetset, robots=robots,DF=DF)

#%% load the simmanger for the folder
# simngr_seqrobot,data_seqrobot = simmanager.SimManager.load("simulations/DynamicSensorTasking-SIMset1/DynamicSensorTasking-SeqExhaust-Robot_0_2021-04-09-14H-44M-44s")
# simngr_seqtime,data_seqtime = simmanager.SimManager.load("simulations/DynamicSensorTasking-SIMset1/DynamicSensorTasking-SeqExhaust-Time_1_2021-04-09-15H-38M-31s")
# simngr_seqtime_MIsearch10,data_seqtime_MIsearch10 = simmanager.SimManager.load("simulations/DynamicSensorTasking-SIMset1/DynamicSensorTasking-SeqExhaust-Time-MIlamda10_0_2021-04-09-15H-18M-40s")
# simngr_seqrobot3,data_seqrobot3 = simmanager.SimManager.load("simulations/DynamicSensorTasking-SIMset1/DynamicSensorTasking-SeqExhaust-Time_1_2021-10-05-20H-16M-43s")
# simngr_seqrobot_MIsearch10,data_seqrobot_MIsearch10 = simmanager.SimManager.load("simulations/DynamicSensorTasking-SIMset1/DynamicSensorTasking-SeqExhaust-Time_0_2021-10-06-12H-36M-21s")

probRatio=[[] for i in  range(len(methods))]
databox_rmsepos=[[] for i in  range(len(methods))]
databox_rmsevel=[[] for i in  range(len(methods))]
databox_tpact=[[] for i in  range(len(methods))]
databox_tpactAfterDisc=[[] for i in  range(len(methods))]
databox_tpactBeforeDisc=[[] for i in  range(len(methods))]

methodstrings=[]
for mi,mth in enumerate(methods):
    mthstr='%s \n($N_s$=%d,$\lambda$=%d)'%mth
    methodstrings.append(mthstr)
    for casecnt in range(10):
        simtestcase = os.path.join(savefolder,"runs","runcase_%s_Ns%02d_L%02d_%03d.pkl"%(mth[0],mth[1],mth[2],casecnt))
        with open(simtestcase,'rb') as F:
            data = pkl.load(F)
            groundtargetset = data['groundtargetset']
            targetset=data['targetset']
            robots=data['robots']
            metrics=data['metrics']
            DF=data['DF']


        for trg in DF:
            probRatio[mi].append(DF[trg]['probRatio'].values)
        
        databox_rmsepos[mi].append(metrics['RMSE_Active_pos'])
        databox_rmsevel[mi].append(metrics['RMSE_Active_vel'])
        databox_tpact[mi].append(metrics['TIMES_Active_percent'])
        databox_tpactAfterDisc[mi].append(metrics['TIMES_Active_percent_after_discovery'])
        databox_tpactBeforeDisc[mi].append(metrics['TIME_percent_before_discover'])
        
        
    probRatio[mi]=np.hstack(probRatio[mi]).astype(np.float64)
    probRatio[mi] = probRatio[mi][~np.isnan(probRatio[mi])]

    databox_rmsepos[mi]=np.hstack(databox_rmsepos[mi]).astype(np.float64)
    databox_rmsepos[mi] = databox_rmsepos[mi][~np.isnan(databox_rmsepos[mi])]
    
    databox_rmsevel[mi]=np.hstack(databox_rmsevel[mi]).astype(np.float64)
    databox_rmsevel[mi] = databox_rmsevel[mi][~np.isnan(databox_rmsevel[mi])]
    
    databox_tpact[mi]=np.hstack(databox_tpact[mi]).astype(np.float64)
    databox_tpact[mi] = databox_tpact[mi][~np.isnan(databox_tpact[mi])]
    
    databox_tpactAfterDisc[mi]=np.hstack(databox_tpactAfterDisc[mi]).astype(np.float64)
    databox_tpactAfterDisc[mi] = databox_tpactAfterDisc[mi][~np.isnan(databox_tpactAfterDisc[mi])]
    
    databox_tpactBeforeDisc[mi]=np.hstack(databox_tpactBeforeDisc[mi]).astype(np.float64)
    databox_tpactBeforeDisc[mi] = databox_tpactBeforeDisc[mi][~np.isnan(databox_tpactBeforeDisc[mi])]
    
    
xpos1=np.arange(len(methodstrings),dtype=np.int)



fig1, ax1 = plt.subplots(figsize=(20, 9))
ax1.set_ylabel('RMSE in position')
ax1.set_xlabel('Methods')
ax1.boxplot(databox_rmsepos,positions = xpos1,boxprops={'linewidth':2},whiskerprops={'linewidth':2})
xticks = ax1.get_xticks()
ax1.set_xticks(xticks)
ax1.set_xticklabels(methodstrings)
# ax1.minorticks_on()
# ax1.tick_params(axis='y', which='minor')
ax1.grid(True,which="both",axis="y")


fig1.savefig(os.path.join(savefolder,"RMSE-BOX-position.png"),format='png',bbox_inches='tight',dpi=600)

# ----------------
fig2, ax2 = plt.subplots(figsize=(20, 9))
ax2.set_ylabel('RMSE in velocity')
ax2.set_xlabel('Methods')
ax2.boxplot(databox_rmsevel,positions = xpos1,boxprops={'linewidth':2},whiskerprops={'linewidth':2})
xticks = ax2.get_xticks()
ax2.set_xticks(xticks)
ax2.set_xticklabels(methodstrings)
ax2.grid(True,which="both",axis="y")

fig2.savefig(os.path.join(savefolder,'RMSE-BOX-velocity.png'),format='png',bbox_inches='tight',dpi=600)


# ----------------

fig3, ax3= plt.subplots(figsize=(20, 9))
ax3.set_ylabel('percent')
ax3.boxplot(databox_tpact,positions = xpos1,boxprops={'linewidth':2},whiskerprops={'linewidth':2})
ax3.set_xticklabels(methodstrings)
ax3.set_ylabel("% Active time steps")
ax3.set_xlabel('Methods')
ax3.grid(True,which="both",axis="y")


fig3.savefig(os.path.join(savefolder,'TIMES_Active_percent.png'),format='png',bbox_inches='tight',dpi=600)



fig4, ax4= plt.subplots(figsize=(20, 9))
ax4.boxplot(databox_tpactAfterDisc,positions = xpos1,boxprops={'linewidth':2},whiskerprops={'linewidth':2})
ax4.set_xticklabels(methodstrings)
ax4.set_ylabel("% Active time steps after discovery")
ax4.set_xlabel('Methods')
ax4.grid(True,which="both",axis="y")

fig4.savefig(os.path.join(savefolder,'TIMES_Active_percent_after_discovery.png'),format='png',bbox_inches='tight',dpi=600)


fig5, ax5= plt.subplots(figsize=(20, 9))
ax5.boxplot(databox_tpactBeforeDisc,positions = xpos1,boxprops={'linewidth':2},whiskerprops={'linewidth':2})
ax5.set_xticklabels(methodstrings)
ax5.set_ylabel("% Time steps before discovery")
ax5.set_xlabel('Methods')
ax5.grid(True,which="both",axis="y")

fig5.savefig(os.path.join(savefolder,'TIME_percent_before_discover.png'),format='png',bbox_inches='tight',dpi=600)

fig6, ax6= plt.subplots(figsize=(20, 9))
ax6.boxplot(probRatio,positions = xpos1,boxprops={'linewidth':2},whiskerprops={'linewidth':2})
ax6.set_xticklabels(methodstrings)
ax6.set_ylabel('p(true)/p(2$\sigma$)')
ax6.set_xlabel('Methods')
ax6.grid(True,which="both",axis="y")


fig6.savefig(os.path.join(savefolder,'probRatio.png'),format='png',bbox_inches='tight',dpi=600)

plt.show()
#%% plot snapshots
# simngr_loaded,data_loaded = simmanager.SimManager.load("simulations/DynamicSensorTasking-SIMset1/DynamicSensorTasking-SeqExhaust-Robot_0_2021-04-09-14H-44M-44s")
simngr_loaded,data_loaded  = simmanager.SimManager.load("simulations/DynamicSensorTasking-SIMset1/DynamicSensorTasking-SeqExhaust-Time_1_2021-04-09-15H-38M-31s")
# simngr_seqtime_MIsearch10,data_seqtime_MIsearch10 = simmanager.SimManager.load("simulations/DynamicSensorTasking-SIMset1/DynamicSensorTasking-SeqExhaust-Time-MIlamda10_0_2021-04-09-15H-18M-40s")
# simngr_seqrobot3,data_seqrobot3 = simmanager.SimManager.load("simulations/DynamicSensorTasking-SIMset1/DynamicSensorTasking-SeqExhaust-Time_1_2021-10-05-20H-16M-43s")
# simngr_seqrobot_MIsearch10,data_seqrobot_MIsearch10 = simmanager.SimManager.load("simulations/DynamicSensorTasking-SIMset1/DynamicSensorTasking-SeqExhaust-Time_0_2021-10-06-12H-36M-21s")

groundtargetset=data_loaded['groundtargetset']
targetset=data_loaded['targetset']
robots=data_loaded['robots']
metrics=data_loaded['metrics']
# tt,status_ku = targetset[1].recorderpost.getvar_uptotime_list('status',t+dt,returntimes=True)


for t,tk,dt in simngr_loaded.iteratetimesteps():
    print (t,tk,dt)
    # update the robots to t+dt
    for j in range(len(robots)):
        # uk = robots[j].getcontrol(t)
        robots[j].xk = robots[j].statehistory[t+dt]
        robots[j].updateTime(t+dt)
        robots[j].updateSensorModel()
    plotsimAllTargs(simngr_loaded,['SimSnapshot'],groundtargetset,targetset,robots,t+dt,simngr_loaded.tvec[:(tk+dt)],saveit=True)
    
    
plt.close("all")    
metrics,DF,figs = utmttrack.multiTarget_tracking_metrics(simngr_loaded.tvec[-1],groundtargetset,targetset)

probRatio=[]
for trg in DF:
    probRatio.append(DF[trg]['probRatio'].values)
probRatio=np.hstack(probRatio)