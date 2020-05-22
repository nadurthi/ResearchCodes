# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import uq.quadratures.cubatures as uqcb
from loggerconfig import *
import numpy as np
from scipy.linalg import block_diag
import collections as clc
import pandas as pd
import os
import dill,pickle

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
# %%

from uq.motfilter import mot as uqmot
from uq.motfilter import jpda
from uq.filters.kalmanfilter import TargetKF
from uq.uqutils.random import genRandomMeanCov
from physmodels.motionmodels import KinematicModel_UM
from physmodels.sensormodels import DiscLTSensorModel
from physmodels import targets as phytarg
import uq.motfilter.measurements as motmeas
from uq.uqutils import recorder as uqrecorder
from uq.uqutils import helper as uqutilhelp
from uq.uqutils import metrics as uqmetrics
from uq.uqutils import simmanager as uqsimmanager

# if __name__=='__main__':
#%%
metalog="""
JPDA test on tracking 5 targets using 1 sensor
this is the initial testing code to see if everything works

Author: Venkat
"""


simmanger = uqsimmanager.SimManager(t0=0,tf=55,dt=0.5,dtplot=0.1,
                                  simname="JPDA_test",savepath="simulations",
                                  workdir=os.getcwd())

simmanger.initialize()




#%%



H = np.hstack((np.eye(2),np.zeros((2,2))))
R = block_diag((1e-1)**2,(1e-1)**2)


filterer = TargetKF()
sensormodel = DiscLTSensorModel(H,R,recorderobj=None,recordSensorState=False)

jpdamotlist={}

filterer = TargetKF()
jpdamot = jpda.JPDAMOT(filterer,recordMeasurementSets=True,PD = 0.8,V=100,uf=None, Gamma=5 )
jpdamot.sensorset.addSensor(sensormodel)
jpdamotlist['KFJPDA_GroundTruthDA']=jpdamot

filterer = TargetKF()
jpdamot = jpda.JPDAMOT(filterer,recordMeasurementSets=True,PD = 0.8,V=100,uf=None, Gamma=15 )
jpdamot.sensorset.addSensor(sensormodel)
jpdamotlist['KFJPDA']=jpdamot

filterer = TargetKF()
jpdamot = jpda.JPDAMOT(filterer,recordMeasurementSets=True,PD = 0.8,V=100,uf=None, Gamma=15 )
jpdamot.sensorset.addSensor(sensormodel)
jpdamotlist['IMMKFJPDA']=jpdamot



vmag = 2

# adding targets
for i in range(5):
    xfk = np.random.rand(4)
    xfk[0:2] = 10*xfk[0:2]
    xfk[2:4] = vmag*xfk[2:4]
    Pfk = np.random.randn(4,4)
    Pfk = np.matmul(Pfk,Pfk.T)

    dynmodel = KinematicModel_UM()

    for jn in jpdamotlist.keys():
        recorderobjprior = uqrecorder.StatesRecorder_fixedDim(statetypes = {
                'xfk':(dynmodel.fn,),'Pfk':(dynmodel.fn,dynmodel.fn)})
        recorderobjpost = uqrecorder.StatesRecorder_fixedDim(statetypes = {
                'xfk':(dynmodel.fn,),'Pfk':(dynmodel.fn,dynmodel.fn)})
        target = phytarg.Target(dynModel=dynmodel, xfk=xfk, Pfk=Pfk, currt = 0, recordfilterstate=True,
                 recorderobjprior = recorderobjprior,recorderobjpost=recorderobjpost)
        target.freeze(recorderobj=False)
        jpdamotlist[jn].targetset.addTarget(target)





#%% get ground truth
for t,tk,dt in simmanger.iteratetimesteps():
    print (t,tk,dt)
    Uk = None
    for jn in jpdamotlist.keys():
        jpdamotlist[jn].propagate(t, dt, Uk)


fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(jpdamotlist['KFJPDA_GroundTruthDA'].targetset.ntargs):
    xfk = jpdamotlist['KFJPDA_GroundTruthDA'].targetset[i].recorderprior.getvar_alltimes('xfk')
    ax.plot(xfk[:,0],xfk[:,1])

plt.show()
#%% Save ground truth and reset filters to new intiial conditions
for i in range(jpdamot.targetset.ntargs):
    for jn in jpdamotlist.keys():
        jpdamotlist[jn].targetset[i].groundtruthrecorder =  jpdamotlist[jn].targetset[i].recorderprior.makecopy()
        jpdamotlist[jn].targetset[i].recorderprior.cleardata(keepInitial = False)
        jpdamotlist[jn].targetset[i].recorderpost.cleardata(keepInitial = False)
        jpdamotlist[jn].targetset[i].defrost(recorderobj=False)

    xf0 = jpdamotlist['KFJPDA_GroundTruthDA'].targetset[i].groundtruthrecorder.getvar_byidx('xfk',0)
    Pf0 = jpdamotlist['KFJPDA_GroundTruthDA'].targetset[i].groundtruthrecorder.getvar_byidx('Pfk',0)
    xf0,_ = genRandomMeanCov(xf0,Pf0,1,np.eye(len(xf0)))

    for jn in jpdamotlist.keys():
        jpdamotlist[jn].targetset[i].setInitialdata(0,xf0, Pf0)

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
    for i in range(jpdamotlist['KFJPDA_GroundTruthDA'].targetset.ntargs):
        xk = jpdamotlist['KFJPDA_GroundTruthDA'].targetset[i].groundtruthrecorder.getvar_bytime('xfk',t+dt)[0]
        ZZ = jpdamotlist['KFJPDA_GroundTruthDA'].sensorset.generateRndMeasSet(t+dt,dt,xk)
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

    jpdamotlist['KFJPDA_GroundTruthDA'].set_DAmat_from_groundtruthDA(t+dt,dt)
    jpdamotlist['KFJPDA'].compute_DAmat(t+dt,dt,Zkset)
    jpdamotlist['IMMKFJPDA'].compute_DAmat(t+dt,dt,Zkset)

    # Zkset should be {'sensID1':[zk1,zk2], 'sensID2':[zk1,zk2,zk3],}
    for jn in jpdamotlist.keys():
        jpdamotlist[jn].measUpdate(t+dt,dt, Zkset)



fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(jpdamotlist['KFJPDA_GroundTruthDA'].targetset.ntargs):
    xfk = jpdamotlist['KFJPDA_GroundTruthDA'].targetset[i].recorderpost.getvar_alltimes('xfk')
    ax.plot(xfk[:,0],xfk[:,1],'--')
    xfk = jpdamotlist['KFJPDA'].targetset[i].recorderpost.getvar_alltimes('xfk')
    ax.plot(xfk[:,0],xfk[:,1])
    xfk = jpdamotlist['IMMKFJPDA'].targetset[i].recorderpost.getvar_alltimes('xfk')
    ax.plot(xfk[:,0],xfk[:,1],'o-')

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(jpdamotlist['KFJPDA'].targetset.ntargs):
    xfk = jpdamotlist['KFJPDA'].targetset[i].recorderpost.getvar_alltimes('xfk')
    ax.plot(xfk[:,0],xfk[:,1])


plt.show()

#%% metrics

jpdametrics = uqmetrics.Metrics()

jpdametrics.Errt=clc.defaultdict(list)
jpdametrics.Rmse=clc.defaultdict(list)

for i in range(jpdamotlist['KFJPDA_GroundTruthDA'].targetset.ntargs):
    for jn in jpdamotlist.keys():
        xt = jpdamotlist[jn].targetset[i].groundtruthrecorder.getvar_alltimes('xfk')
        xf = jpdamotlist[jn].targetset[i].recorderpost.getvar_alltimes('xfk')

        errt,rmse = uqmetrics.getEsterror(xt,xf,
                                          stateset={'state':[0,1,2,3],
                                                    'pos':[0,1],
                                                    'vel':[2,3]
                                                    })
        jpdametrics.Errt[jn].append( errt )
        jpdametrics.Rmse[jn].append( rmse )


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(simmanger.tvec,jpdametrics.Errt['KFJPDA_GroundTruthDA'][0]['state'],label='GTDA')
ax.plot(simmanger.tvec,jpdametrics.Errt['KFJPDA'][0]['state'],label='KFJPDA')
ax.plot(simmanger.tvec,jpdametrics.Errt['IMMKFJPDA'][0]['state'],label='IMMKFJPDA')
ax.legend()
plt.show()
plt.pause(0.1)
simmanger.savefigure(fig,['post','K'],'state.png')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(simmanger.tvec,jpdametrics.Errt['KFJPDA_GroundTruthDA'][0]['pos'])
plt.pause(0.1)
simmanger.savefigure(fig,['post','K'],'pos.png')




plt.show()
#%% Saving
simmanger.finalize()

simmanger.save(metalog,mainfile=__file__, jpdametrics=jpdametrics, jpdamot=jpdamot )

debugStatuslist = jpdamot.debugStatus()
uqutilhelp.DebugStatus().writestatus(debugStatuslist,simmanger.debugstatusfilepath)






