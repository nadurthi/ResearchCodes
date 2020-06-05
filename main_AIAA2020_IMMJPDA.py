"""IMM+JPDA main function for testinf and simulation."""

import matplotlib
import numpy as np
from scipy.linalg import block_diag
import collections as clc
import os
from uq.motfilter import jpda
import uq.filters.kalmanfilter as uqkf
from uq.filters import sigmafilter as uqsigf
from uq.uqutils.random import genRandomMeanCov
from physmodels import motionmodels as phymm
from physmodels import sensormodels as physm
from physmodels import targets as phytarg
from uq.filters import imm as immfilter
from uq.filters import gmmbase as uqfgmmbase
from uq.uqutils import recorder as uqrecorder
from uq.uqutils import metrics as uqmetrics
from uq.uqutils import simmanager as uqsimmanager
from uq.quadratures import cubatures as quadcub
import uq.quadratures.cubatures as uqcb
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from loggerconfig import *


plt.close('all')

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
AIAA SCITECH 2020 IMM+JPDA paper simulations
Author: Venkat
Date: June 4 2020

Comparing the IMM+JPDA filter using EKF and UT and CUT points
"""


simmanger = uqsimmanager.SimManager(t0=0,tf=55,dt=0.5,dtplot=0.1,
                                  simname="AIAA-SCITECH-2020-JPDA-IMM",savepath="simulations",
                                  workdir=os.getcwd())

simmanger.initialize()




#%%



# H = np.hstack((np.eye(2),np.zeros((2,2))))
# R = block_diag((0.5)**2,(0.5)**2)
# sensormodel4d = physm.DiscLTSensorModel(H,R,recorderobj=None,recordSensorState=False)

H = np.hstack((np.eye(2), np.zeros((2, 3))))
R = block_diag((0.5)**2, (0.5)**2)
sensormodel5d = physm.DiscLTSensorModel(H, R, recorderobj=None, recordSensorState=False)
sensorset = physm.SensorSet()
sensorset.addSensor(sensormodel5d)
jpdamotlist = {}

# filterer = TargetKF()
# jpdamot = jpda.JPDAMOT(filterer,recordMeasurementSets=True,PD = 0.7,V=100,uf=None, Gamma=5 )
# jpdamot.sensorset.addSensor(sensormodel4d)
# jpdamotlist['EKFJPDA_GroundTruthDA']=jpdamot

filterer = uqkf.TargetKF()
jpdamot = jpda.JPDAMOT(filterer,recordMeasurementSets=False,PD = 0.7,V=100,uf=None, Gamma=15 )
jpdamot.sensorset = sensorset
jpdamotlist['EKFJPDA']=jpdamot

filterer = uqsigf.TargetSigmaKF(sigmamethod = quadcub.SigmaMethod.UT)
jpdamot = jpda.JPDAMOT(filterer,recordMeasurementSets=False,PD = 0.7,V=100,uf=None, Gamma=15 )
jpdamot.sensorset = sensorset
jpdamotlist['UKFJPDA']=jpdamot

modefilterer = uqkf.KFfilterer()
filterer = immfilter.TargetIMM(modefilterer)
jpdamot = jpda.JPDAMOT(filterer,recordMeasurementSets=False,PD = 0.7,V=100,uf=None, Gamma=15 )
jpdamot.sensorset = sensorset
jpdamotlist['EKFIMMJPDA']=jpdamot

modefilterer = uqsigf.Sigmafilterer(sigmamethod = quadcub.SigmaMethod.UT)
filterer = immfilter.TargetIMM(modefilterer)
jpdamot = jpda.JPDAMOT(filterer,recordMeasurementSets=False,PD = 0.7,V=100,uf=None, Gamma=15 )
jpdamot.sensorset = sensorset
jpdamotlist['UKFIMMJPDA']=jpdamot


groundTruthTargets = phytarg.TargetSet()

vmag = 2

# adding targets
for i in range(5):

    xfk = np.random.rand(4)
    xfk[0:2] = 25*xfk[0:2]
    xfk[2:4] = vmag*xfk[2:4]
    Pfk = np.random.randn(4,4)
    Pfk = np.matmul(Pfk,Pfk.T)
    xfk5 = np.hstack([xfk,np.random.rand()])
    Pfk5 = block_diag(Pfk,0.1**2)
            
    dynmodel = phymm.KinematicModel_UM_5state()
    
    um1 = phymm.KinematicModel_UM_5state()
    ct2 = phymm.KinematicModel_CT()
    p=np.array([[0.90,0.1],[0.1,0.90]])
    dynMultiModels = phymm.MultipleMarkovMotionModel([um1,ct2],p)
    
    target = phytarg.Target(dynModel=dynmodel, xfk=xfk5, Pfk=Pfk5, currt = 0, recordfilterstate=True,
             recorderobjprior = None,recorderobjpost=None)
    target.groundtruthrecorder = uqrecorder.StatesRecorder_fixedDim(statetypes = {
            'xfk':(dynmodel.fn,)})
        
    groundTruthTargets.addTarget(target)
    
    for jn in jpdamotlist.keys():
        if 'IMM' in jn:
            
            gmm0 = uqfgmmbase.GMM.fromlist([xfk5,xfk5],[Pfk5,Pfk5],[0.5,0.5],0)
            modelprob0=np.array([0.5,0.5])
            recorderobjprior = uqrecorder.StatesRecorder_list(statetypes = ['xfk','Pfk','gmmfk','modelprobfk'] )
            recorderobjpost = uqrecorder.StatesRecorder_list(statetypes = ['xfk','Pfk','gmmfk','modelprobfk'] )
            target = phytarg.Target(dynModelset=dynMultiModels, gmmfk=gmm0, modelprobfk = modelprob0, currt = 0, recordfilterstate=True,
                     recorderobjprior = recorderobjprior,recorderobjpost=recorderobjpost)

            target.groundtruthrecorder = uqrecorder.StatesRecorder_fixedDim(statetypes = {
                    'xfk':(ct2.fn,)})
            m,P=gmm0.weightedest(modelprob0)
#            target.setInitialdata(0,gmmfk=gmm0, modelprobfk=modelprob0,xfk=m,Pfk=P)

        else:
            recorderobjprior = uqrecorder.StatesRecorder_fixedDim(statetypes = {
                    'xfk':(dynmodel.fn,),'Pfk':(dynmodel.fn,dynmodel.fn)})
            recorderobjpost = uqrecorder.StatesRecorder_fixedDim(statetypes = {
                    'xfk':(dynmodel.fn,),'Pfk':(dynmodel.fn,dynmodel.fn)})
            target = phytarg.Target(dynModel=dynmodel, xfk=xfk, Pfk=Pfk, currt = 0, recordfilterstate=True,
                     recorderobjprior = recorderobjprior,recorderobjpost=recorderobjpost)
#            target.setInitialdata(0,xfk=xfk, Pfk=Pfk)

        target.freeze(recorderobj=False)
        jpdamotlist[jn].targetset.addTarget(target)





#%% get ground truth

for n in range(groundTruthTargets.ntargs):
    xk = groundTruthTargets[n].xfk.copy()
    groundTruthTargets[n].groundtruthrecorder.record(0,xfk=xk)
    ss = np.sign(np.random.randn(1)[0])
    nt = simmanger.ntimesteps
    for t,tk,dt in simmanger.iteratetimesteps():
        if 0<=tk<20:
            xk[4]=0
        if 20<=tk<30:
            xk[4]=0.5*ss
        if 30<=tk<50:
            xk[4]=0
        if 50<=tk<60:
            xk[4]=-0.5*ss
        if 60<=tk<80:
            xk[4]=0
        if 80<=tk<100:
            xk[4]=0.3*ss

        _,xk = ct2.propforward( t, dt, xk, uk=0)
        groundTruthTargets[n].groundtruthrecorder.record(t+dt,xfk=xk)



fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(groundTruthTargets.ntargs):
    xfk = groundTruthTargets[i].groundtruthrecorder.getvar_alltimes('xfk')
    ax.plot(xfk[:,0],xfk[:,1])

plt.show()
#%% Save ground truth and reset filters to new intiial conditions
for i in range(groundTruthTargets.ntargs):

    xf0 = groundTruthTargets[i].groundtruthrecorder.getvar_byidx('xfk',0)
#    Pf0 = jpdamotlist['IMMKFJPDA'].targetset[i].groundtruthrecorder.getvar_byidx('Pfk',0)
    Pf0 = groundTruthTargets[i].Pfk.copy()
    xf0,_ = genRandomMeanCov(xf0,Pf0,0.8,np.eye(len(xf0)))

    for jn in jpdamotlist.keys():
        if 'IMM' in jn:
            gmm0 = uqfgmmbase.GMM.fromlist([xf0,xf0],[Pf0,Pf0],[0.5,0.5],0)
            modelprobf0 = np.array([0.5,0.5])
            m,P=gmm0.weightedest(modelprob0)
            jpdamotlist[jn].targetset[i].setInitialdata(0,gmmfk=gmm0, modelprobfk=modelprobf0,xfk=m,Pfk=P)
        else:
            jpdamotlist[jn].targetset[i].setInitialdata(0,xfk=xf0, Pfk=Pf0)


#%% Run the filter
for t,tk,dt in simmanger.iteratetimesteps():
    print (t,tk,dt)
    Uk = None
    for jn in jpdamotlist.keys():
        jpdamotlist[jn].propagate(t, dt, Uk)
    # after prop, the time is t+dt

    # generate a random measurement
    Zkset = clc.defaultdict(list)


    grounttruthDA= {}

    # generate measurements and ground truth
    for i in range(groundTruthTargets.ntargs):
        xk = groundTruthTargets[i].groundtruthrecorder.getvar_bytime('xfk',t+dt)[0]
        ZZ = sensorset.generateRndMeasSet(t+dt,dt,xk)
        # ZZ = {'ID': {'zk': zk, 'isinsidek':isinsidek, 'Lk': Lk   } }
        for sensID in ZZ:
            Zkset[sensID].append( ZZ[sensID]['zk'] )
            if sensID not in grounttruthDA:
                grounttruthDA[sensID] = np.zeros((jpdamot.targetset.ntargs,1))
            n=len(Zkset[sensID])
            grounttruthDA[sensID]=np.hstack([grounttruthDA[sensID], np.zeros((jpdamot.targetset.ntargs,1))])
            grounttruthDA[sensID][i,n]=1

    for jn in jpdamotlist.keys():
        jpdamotlist[jn].recordermeas.record(t+dt,Zk=Zkset)

    for jn in jpdamotlist.keys():
        jpdamotlist[jn].setgrounttruthDA(t+dt,dt, grounttruthDA )
        
    for jn in jpdamotlist.keys():
        jpdamotlist[jn].compute_DAmat(t+dt,dt,Zkset)

    # Zkset should be {'sensID1':[zk1,zk2], 'sensID2':[zk1,zk2,zk3],}
    for jn in jpdamotlist.keys():
        jpdamotlist[jn].measUpdate(t+dt,dt, Zkset)


markers = ['--','-','o-','s-','k']
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(groundTruthTargets.ntargs):
    for idx,jn in enumerate(jpdamotlist.keys()):
        xfk = jpdamotlist[jn].targetset[i].recorderpost.getvar_alltimes('xfk')
        ax.plot(xfk[:,0],xfk[:,1],markers[idx])


plt.show()

# %% metrics

jpdametrics = uqmetrics.Metrics()

jpdametrics.Errt = clc.defaultdict(list)
jpdametrics.Rmse = clc.defaultdict(list)

for i in range(groundTruthTargets.ntargs):
    xt = groundTruthTargets[i].groundtruthrecorder.getvar_alltimes('xfk')
    for jn in jpdamotlist.keys():
        xf = jpdamotlist[jn].targetset[i].recorderpost.getvar_alltimes('xfk')
        errt, rmse = uqmetrics.getEsterror(xt, xf, stateset={
                                              'state': [0, 1, 2, 3],
                                              'pos': [0, 1],
                                              'vel': [2, 3]
                                                    })
        jpdametrics.Errt[jn].append(errt)
        jpdametrics.Rmse[jn].append(rmse)


fig = plt.figure()
ax = fig.add_subplot(111)
for idx,jn in enumerate(jpdamotlist.keys()):
    ax.plot(simmanger.tvec, jpdametrics.Errt[jn][0]['state'], label=jn)

ax.legend()
plt.show()
plt.pause(0.1)
simmanger.savefigure(fig, ['post', 'K'], 'state.png')

fig = plt.figure()
ax = fig.add_subplot(111)
for idx,jn in enumerate(jpdamotlist.keys()):
    ax.plot(simmanger.tvec, jpdametrics.Errt[jn][0]['state'], label=jn)
    
plt.pause(0.1)
simmanger.savefigure(fig, ['post', 'K'], 'pos.png')

plt.show()
# %% Saving
simmanger.finalize()

simmanger.save(metalog, mainfile=__file__, jpdametrics=jpdametrics, jpdamot=jpdamot)

# debugStatuslist = jpdamot.debugStatus()
# uqutilhelp.DebugStatus().writestatus(debugStatuslist,simmanger.debugstatusfilepath)
