
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
from sensortasking import robotdp as robdb
# %% script-level properties

runfilename = __file__
metalog="""
Journal paper on Dynamic DP sensor tasking paper simulations. Mostly submitted to TAES
Author: Venkat
Date: June 4 2020

Comparing the IMM+JPDA filter using EKF and UT and CUT points
"""


simmanger = uqsimmanager.SimManager(t0=0,tf=200,dt=2,dtplot=0.1,
                                  simname="DynamicSensorTasking-DP-Template",savepath="simulations",
                                  workdir=os.getcwd())

simmanger.initialize()


# %% Create Map

rgmap = gridmap.Regular2DNodeGrid(xy0=(0,0),xyf=(101,101),d=(10,10))
fig = plt.figure()
ax = fig.add_subplot(111)
rgmap.plotmap(ax)
plt.pause(0.1)



# %% Set UAVs

robotTempTrajsSavedFile = "robot/RobotMapData/Main2DRegGrid_1.pkl"
if os.path.isfile(robotTempTrajsSavedFile):
    with open(robotTempTrajsSavedFile,"rb") as FF:
        robot=pkl.load(FF)
else:
    robot = robtgr.Robot2DRegGrid()
    robot.dynModel = phymm.KinematicModel_CT_control()
    robot.mapobj = rgmap
    
    robot.xk=np.array([5,5,np.pi/2])
    
    # H = np.hstack((np.eye(2), np.zeros((2, 3))))
    # R = block_diag((1)**2, (2*np.pi/180)**2)
    # robot.sensormodel = physmfov.RTHcircularFOVsensor(xc=np.array([0,0]), R=R, posstates=[0,1], enforceConstraint=True,
                 # FOVradius=1, FOVsectorhalfangle=1, dirn=1,recordSensorState=False)
    
    robot.MaxVmag = 4 #5
    robot.MaxOmega = 2.5 #3
    robot.NominalVmag = 2 #2
    robot.MinTempXferr = 0.5
    
    with utltm.TimingContext():
        robttj.generateTemplates(robot,min([0.5,simmanger.dt/20]),simmanger.dt)

    with open("robot/RobotMapData/Main2DRegGrid_1.pkl","wb") as FF:
        pkl.dump(robot,FF,protocol=pkl.HIGHEST_PROTOCOL)


robot.mapobj.xy0 =np.array(robot.mapobj.xy0)
robot.mapobj.xyf =np.array(robot.mapobj.xyf)

# plotting templates
fig = plt.figure()
ax = fig.add_subplot(111)
xnode0 = np.array([50,50])
th0 = np.pi
for uk in robot.iterateControls(xnode0,th0):
    ax.plot(uk['Xtraj'][:,0],uk['Xtraj'][:,1],'r')
plt.show()

# now setting the robots
robots=[robtgr.Robot2DRegGrid(),robtgr.Robot2DRegGrid()]



robots[0].robotName= 'UAV:0'
robots[0].dynModel = phymm.KinematicModel_CT_control(L1=0.016, L2=0.0001)
robots[0].mapobj = robot.mapobj
robots[0].controltemplates = copy.deepcopy(robot.controltemplates)
robots[0].xk=np.array([70,70,np.pi/2])
robots[0].robotColor = 'b'
robots[0].sensormodel=physmfov.RTHcircularFOVsensor(R=block_diag((1)**2, (0.5*np.pi/180)**2), 
                                                    FOVradius=20,
                                                    FOVcolor='b',
                                                    TP=0.9, TN=0.9, FP=0.1, FN=0.1,
                                                    recordSensorState=True,
                                                    enforceConstraint=True)
robots[0].updateSensorModel()


robots[1].robotName= 'UAV:1'
robots[1].dynModel = phymm.KinematicModel_CT_control(L1=0.016, L2=0.0001)
robots[1].mapobj = robot.mapobj
robots[1].controltemplates = copy.deepcopy(robot.controltemplates)
robots[1].xk=np.array([30,30,np.pi/2])
robots[1].robotColor = 'r'
robots[1].sensormodel=physmfov.RTHcircularFOVsensor(R=block_diag((1)**2, (0.5*np.pi/180)**2), 
                                                    FOVradius=15,
                                                    FOVcolor='r',
                                                    TP=0.9, TN=0.9, FP=0.1, FN=0.1,
                                                    recordSensorState=True,
                                                    enforceConstraint=True)
robots[1].updateSensorModel()   



robots.pop(1)
# %% Create Targets
targetset = phytarg.TargetSet()
vmag = 1
DD = []
flname = "testcases/target.pkl"
if os.path.isfile(flname):
    with open(flname,'rb') as F:
        DD=pkl.load(F)

for i in range(2):
    if len(DD)>i:
        xf0 = DD[i]['xf0']
        Pf0 = DD[i]['Pf0']
    else:
        xf0 = np.random.rand(5)
        xf0[0:2] = 100*xf0[0:2]
        xf0[2:4] = np.random.randn(2)
        xf0[2:4] = vmag*xf0[2:4]/nplg.norm(xf0[2:4])
        # xf0[2:4] = vmag*xf0[2:4]
        xf0[4] = 0.01*np.random.randn()
        Pf0 = np.random.randn(5,5)
        Pf0 = np.matmul(Pf0,Pf0.T)
        Pf0[0:2,0:2]=10*Pf0[0:2,0:2]/np.max(Pf0[0:2,0:2].reshape(-1))
        Pf0[2:4,2:4]=0.1*Pf0[2:4,2:4]/np.max(Pf0[2:4,2:4].reshape(-1))
        Pf0[4,4]=0.0001
        
        # if i==0:
        #     Pf0[0:2,0:2]=15*Pf0[0:2,0:2]
        DD.append({'xf0':xf0,'Pf0':Pf0})
        
    gmm0 = uqgmmbase.GMM.fromlist([xf0],[Pf0],[1],0)
    
    dynmodel = phymm.KinematicModel_CT_control(L1=1e-5, L2=1e-8)

    recorderobjprior = uqrecorder.StatesRecorder_list(statetypes = ['gmmfk','xfk','Pfk'])
    recorderobjpost = uqrecorder.StatesRecorder_list(statetypes = ['gmmfk','xfk','Pfk'])
    target = phytarg.Target(dynModel=dynmodel, gmmfk=gmm0, currt = 0, recordfilterstate=True,
                 recorderobjprior = recorderobjprior,recorderobjpost=recorderobjpost)
    target.groundtruthrecorder = uqrecorder.StatesRecorder_list(statetypes = ['xtk','uk'] )
    target.setInitialdata(0,gmmfk=gmm0)
    target.freeze(recorderobj=False)
    target.groundtruthrecorder.record(0, xtk=xf0,uk=None)
    
    target.targetName = "Trgt:%d"%i
    targetset.addTarget(target)
        
# search map using multiple Gaussians: search-track

# create ground truth
if not os.path.isfile(flname):
    with open(flname,'wb') as F:
        pkl.dump(DD,F)

# %% plotting functions
def plotsimIndividTargs(simmanger,targetset,robots,t):
    print("plotsim thread")
    for i in range(targetset.ntargs):
        fig = plt.figure("MainSim-Contour: Target: %d"%i)
        axlist = fig.axes
        if len(axlist)==0:
            ax = fig.add_subplot(111,label='contour')
        else:
            ax = axlist[0]
            ax.cla()
        
        ax.set_title("time step = %f"%(t,) )
        robots[0].mapobj.plotmap(ax)
        for r in robots:
            r.plotrobot(ax)
            
        xktruth = targetset[i].groundtruthrecorder.getvar_uptotime_stacked('xtk',t)
        gmmu = targetset[i].recorderpost.getvar_bytime('gmmfk',t)
        if gmmu is None:
            gmmu = targetset[i].recorderprior.getvar_bytime('gmmfk',t)
            
        gmmupos = uqgmmbase.marginalizeGMM(gmmu,targetset[i].posstates)
        XX = uqgmmbase.plotGMM2Dcontour(gmmupos,nsig=1,N=100,rettype='list')
        for cc in range(gmmupos.Ncomp):
            ax.plot(XX[cc][:,0],XX[cc][:,1],c=colors[i])
            ax.annotate("wt: "+str(gmmupos.w(cc))[:5],gmmupos.m(cc)[0:2],gmmupos.m(cc)[0:2]-3,color=colors[i],fontsize='x-small')
            
        ax.plot(xktruth[:,0],xktruth[:,1],linestyle='--',c=colors[i])
        ax.plot(xktruth[-1,0],xktruth[-1,1],c=colors[i],marker='*')
        ax.annotate(targetset[i].targetName,xktruth[-1,0:2],xktruth[-1,0:2]+2,color=colors[i],fontsize='x-small')
    
    # 
        ax.axis('equal')
        ax.set(xlim=(-15, 115), ylim=(-15, 115))
        plt.show(block=False)
        plt.pause(0.4)
        # k = f'{t:06.2f}'.replace('.','-')
        # simmanger.savefigure(fig, ['SimSnapshot', 'targ',], k+'.png')

def plotsimAllTargs(simmanger,targetset,robots,t,tvec):
    print("plotsim thread")
    fig = plt.figure("MainSim-Contour-all targs")
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
        
            
        xktruth = targetset[i].groundtruthrecorder.getvar_uptotime_stacked('xtk',t)
        gmmu = targetset[i].recorderpost.getvar_bytime('gmmfk',t)
        if gmmu is None:
            gmmu = targetset[i].recorderprior.getvar_bytime('gmmfk',t)
            
        gmmupos = uqgmmbase.marginalizeGMM(gmmu,targetset[i].posstates)
        XX = uqgmmbase.plotGMM2Dcontour(gmmupos,nsig=1,N=100,rettype='list')
        for cc in range(gmmupos.Ncomp):
            ax.plot(XX[cc][:,0],XX[cc][:,1],c=colors[i])
            ax.annotate("wt: "+str(gmmupos.w(cc))[:5],gmmupos.m(cc)[0:2],gmmupos.m(cc)[0:2]-3,color=colors[i],fontsize='x-small')
            
        ax.plot(xktruth[:,0],xktruth[:,1],linestyle='--',c=colors[i])
        ax.plot(xktruth[-1,0],xktruth[-1,1],c=colors[i],marker='*')
        ax.annotate(targetset[i].targetName,xktruth[-1,0:2],xktruth[-1,0:2]+2,color=colors[i],fontsize='x-small')
    
    # 
    ax.axis('equal')
    ax.set(xlim=(-15, 115), ylim=(-15, 115))
    plt.show(block=False)
    plt.pause(0.4)
    k = f'{t:06.2f}'.replace('.','-')
    simmanger.savefigure(fig, ['SimSnapshot', 'Alltargs'], k+'.png',data=[targetset,robots,t,tvec])
    
# %% Run sim
NUM_COLORS = 20

colors = [colmap(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
shuffle(colors)

Targetfilterer = uqgmmf.TargetGMM(modefilterer=uqfsigf.Sigmafilterer( sigmamethod=quadcub.SigmaMethod.UT) )
mapobj = robot.mapobj

gtruth_turnrateMax = 0.25
gtruth_turnrateChangerate=0.8

mergerConfigDP = uqgmmmerg.MergerConfig(meanabs=0.5,meanthresfrac=0.5,dh=0.7,
                                      wtthreshprune=1e-3,doMerge=True,
                                      alorithm="EMfixedComps",fixedComps=5,fixedCompsNmc=2000)
splitterConfigDP = uqgmmsplit.SplitterConfig(alphaSpread=2,Ngh=5,nsig=2,Nmc=1000,
                                           NcompIN=3,NcompOUT=3,minfrac=5,
                                           wtthreshprune=1e-3)

mergerConfigMeasUpdt = uqgmmmerg.MergerConfig(meanabs=0.5,meanthresfrac=0.5,dh=0.7,
                                      wtthreshprune=1e-3,doMerge=True,
                                      alorithm="EMfixedComps",fixedComps=5,fixedCompsNmc=2000)
splitterConfigMeasUpdt = uqgmmsplit.SplitterConfig(alphaSpread=2,Ngh=5,nsig=2,Nmc=1000,
                                           NcompIN=3,NcompOUT=3,minfrac=5,
                                           wtthreshprune=1e-3)


class InfoConfig:
    def __init__(self):
        self.bins = (tuple(np.linspace(0,100,30)),
                     tuple(np.linspace(0,100,30)),
                     tuple(np.linspace(-5,5,20)),
                     tuple(np.linspace(-5,5,20)),
                     tuple(np.linspace(-1,1,20)))
        self.binmaxwidth = [3,3,1,1,0.5]
        
infoconfig = InfoConfig()

axCont = {}
        
TT = 5
TTrecomp =2
dptvec = [0]

for t,tk,dt in simmanger.iteratetimesteps():
    print (t,tk,dt)
    
    if True:
        for nt in range(targetset.ntargs):
            gmmupos = uqgmmbase.marginalizeGMM(targetset[nt].gmmfk,targetset[nt].posstates)
            m,P = gmmupos.meanCov()
            u,v = nplg.eig(P)
            print(u,v)
            gg=np.all([robots[0].mapobj.isInMap(xk) for xk in gmmupos.mu])
            Xmc = gmmupos.random(1000)
            gg=np.sum([robots[0].mapobj.isInMap(xk) for xk in Xmc])
            if np.max(np.sqrt(u))>50 or gg/1000 <= 0.5:
                Nc = 5
                clf = mixture.GaussianMixture(n_components=5, covariance_type='full')
                clf.fit(robots[0].mapobj.XYgvec)
                gmmuk = uqgmmbase.GMM(None,None,None,0)
                m = np.hstack([clf.means_,np.zeros((Nc,3))])
                p=[]
                for i in range(Nc):
                    p.append(block_diag(clf.covariances_[i],0.0001*np.diag((1,1)),0.00001 ))
                p = np.stack(p,axis=0)
                # gmmuk.appendCompArray(clf.means_,np.stack([clf.covariances_]*mergerConfig.fixedComps,axis=0),clf.weights_,renormalize = True)
                gmmuk.appendCompArray(m,p,clf.weights_,renormalize = True)
                targetset[nt].gmmfk = gmmuk.makeCopy()
                print('Target ',nt," reset at t = ",t)
    # optimize the control trajectories
    
    if t==0 or t==dptvec[TTrecomp]:
        dptvec = simmanger.tvec[tk:min([tk+TT,simmanger.ntimesteps]) ]
        robdb.dynamicProgRobot2Dgrid_SeqRobot(simmanger,dptvec,robots,targetset,Targetfilterer,infoconfig,splitterConfigDP,mergerConfigDP)

    Uk = []
    # generate truth at time t+dt
    for i in range(targetset.ntargs):
        xk = targetset[i].groundtruthrecorder.getvar_bytime('xtk',t)
        vmag = nplg.norm(xk[2:4])
        if vmag >0:
            vdir = xk[2:4]/vmag
        else:
            vdir = np.array([0,0])
        
        if not mapobj.isInMap(xk):
            dirnmid = mapobj.middirn()-xk[0:2]
            dirnmid = dirnmid/nplg.norm(dirnmid)
            c=np.cross(np.hstack([vdir,0]),np.hstack([dirnmid,0]))
            turnrate = np.sign(c[2])*np.arccos(np.dot(vdir,dirnmid))/(simmanger.dt*5)
            uk = [vmag,turnrate]
        else:
            # random turn of target
            if np.random.rand()>gtruth_turnrateChangerate:
                turnrate = xk[4]+max([min([gtruth_turnrateMax*np.random.randn(),gtruth_turnrateMax]),-gtruth_turnrateMax])
                uk = [vmag,turnrate]
            else:
                uk = [vmag,0]
                
        Uk.append(uk)
        # print("uk = ",uk)
        _,xk1=targetset[i].dynModel.propforward( t, dt, xk, uk=uk)
        targetset[i].groundtruthrecorder.recordupdate(t+dt,xtk=xk1,uk=uk)
    
    # update the robots to t+dt
    for j in range(len(robots)):
        # uk = robots[j].getcontrol(t)
        robots[j].xk = robots[j].statehistory[t+dt]
        robots[j].updateTime(t+dt)
        robots[j].updateSensorModel()
        
        
    
    Ukfilter = [None]*targetset.ntargs
    # propagate the targets
    for i in range(targetset.ntargs):
        Targetfilterer.propagate(t,dt,targetset[i],Ukfilter[i],updttarget=True)
    
    # generate measurements
    Zk=clc.defaultdict(list)
    for i in range(targetset.ntargs):
        xk = targetset[i].groundtruthrecorder.getvar_bytime('xtk',t+dt)
        for j in range(len(robots)):
            zij,isinsidek,Lk = robots[j].sensormodel.generateRndMeas(t+dt, dt, xk,useRecord=False)
            if isinsidek == 0:
                Zk[i].append(None)
            else:
                Zk[i].append(zij)
                print("Target %d is in FOV of UAV %d at time %f or timestep %d"%(i,j,t+dt,tk+1))
                
    # do measurment update for targets
    with utltm.TimingContext("Measureemt update: "):
        for i in range(targetset.ntargs):
            n = targetset[i].gmmfk.Ncomp
            # measUpdateFOV_PFGMM
            # measUpdateFOV_randomsplitter
            rbf2df.measUpdateFOV_randomsplitter(t+dt,dt,robots,targetset[i],Zk[i],Targetfilterer,infoconfig,
                                      updttarget=True,
                                      splitterConfig = splitterConfigMeasUpdt,
                                      mergerConfig = mergerConfigMeasUpdt,computePriorPostDist=False)
            print("Target %d: #components before: %d , after: %d"%(i,n,targetset[i].gmmfk.Ncomp))
    
    # Plotting of simulations
    # thrd=threading.Thread(name="sim-plot-thread",target=plotsim,args=(targetset,robots,t+dt))
    # thrd.setDaemon(True)
    # thrd.start()
    # thrd.join()
    # plotsim(targetset,robots,t+dt)
    plotsimIndividTargs(simmanger,targetset,robots,t+dt)
    plotsimAllTargs(simmanger,targetset,robots,t+dt,simmanger.tvec[:(tk+dt)])
    # # plotting surface
    # fig = plt.figure("MainSim-Surf")
    # ax = fig.add_subplot(111, projection='3d')
    # plt.title("time step = %d"%tk)
    # robots[0].mapobj.plotmap(ax)
    # for r in robots:
    #     r.plotrobot(ax)
    # for i in range(targetset.ntargs):
    #     xktruth = targetset[i].groundtruthrecorder.getvar_bytime('xtk',t)
    #     gmmu = targetset[i].recorderpost.getvar_bytime('gmmfk',t)
    #     gmmupos = uqgmmbase.marginalizeGMM(gmmu,targetset[i].posstates)
    #     xx,yy,p = uqgmmbase.plotGMM2Dsurf(gmmupos,Ng=50)
    #     ax.plot_surface(xx,yy,p,color=colors[i],alpha=0.6,linewidth=1) 
    
    # plt.pause(0.1)
    # plt.show()
    plt.ion()
    plt.show()
    plt.pause(0.1)
    

# for t,tk,dt in simmanger.iteratetimesteps():
#     fig = plt.figure("Info-Surf: %f"%t)
#     simmanger.savefigure(fig, ['SimSnapshot', 'InfoMap'], str(int(tk))+'.png')
# %% Finalize and save

simmanger.finalize()

simmanger.save(metalog, mainfile=runfilename, targetset=targetset, robots=robots,
               mergerConfigDP=mergerConfigDP,splitterConfigDP=splitterConfigDP,
               mergerConfigMeasUpdt=mergerConfigMeasUpdt,splitterConfigMeasUpdt=splitterConfigMeasUpdt,
               TT=TT,TTrecomp=TTrecomp,gtruth_turnrateMax=gtruth_turnrateMax,
               gtruth_turnrateChangerate=gtruth_turnrateChangerate,vmag=vmag,flname=flname,
               robotTempTrajsSavedFile=robotTempTrajsSavedFile)

# simmanger.save(metalog, mainfile=runfilename)


