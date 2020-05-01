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
from uq.motfilter import targets as uqtargets
from uq.uqutils.random import genRandomMeanCov
from physmodels.motionmodels import KinematicModel_UM
from physmodels.sensormodels import DiscLTSensorModel
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

filterer = TargetKF()

H = np.hstack((np.eye(2),np.zeros((2,2))))
R = block_diag((1e-1)**2,(1e-1)**2)



sensormodel = DiscLTSensorModel(H,R,recorderobj=None,recordSensorState=True)

jpdamot = jpda.JPDAMOT(filterer,recordMeasurementSets=True)
jpdamot.sensorset.addSensor(sensormodel)





vmag = 2

# adding targets
for i in range(5):
    xfk = np.random.rand(4)
    xfk[0:2] = 10*xfk[0:2]
    xfk[2:4] = vmag*xfk[2:4]
    Pfk = np.random.randn(4,4)
    Pfk = np.matmul(Pfk,Pfk.T)

    dynmodel = KinematicModel_UM()


    recorderobjprior = uqrecorder.StatesRecorder_fixedDim(statetypes = {
            'xfk':(dynmodel.fn,),'Pfk':(dynmodel.fn,dynmodel.fn)})
    recorderobjpost = uqrecorder.StatesRecorder_fixedDim(statetypes = {
            'xfk':(dynmodel.fn,),'Pfk':(dynmodel.fn,dynmodel.fn)})
    target = uqtargets.Target(dynModel=dynmodel, xfk=xfk, Pfk=Pfk, currtk = 0, recordfilterstate=True,
            status='active', recorderobjprior = recorderobjprior,recorderobjpost=recorderobjpost,
            filterer=filterer,saveinitializingdata=True)

    jpdamot.targetset.addTarget(target)





#%% get ground truth
for t,tk,dt in simmanger.iteratetimesteps():
    print (t,tk,dt)
    Uk = None
    jpdamot.propagate(t, dt, Uk)

# jpdamot.filterer.debugStatus()
debugStatuslist = jpdamot.debugStatus()
uqutilhelp.DebugStatus().writestatus(debugStatuslist,simmanger.debugstatusfilepath)

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(jpdamot.targetset.ntargs):
    xfk = jpdamot.targetset[i].recorderprior.getvar_alltimes('xfk')
    ax.plot(xfk[:,0],xfk[:,1])

plt.show()
#%% Save ground truth and reset filters to new intiial conditions
for i in range(jpdamot.targetset.ntargs):
    jpdamot.targetset[i].groundtruthrecorder=  jpdamot.targetset[i].recorderprior.makecopy()
    jpdamot.targetset[i].recorderprior.cleardata(keepInitial = False)
    jpdamot.targetset[i].recorderpost.cleardata(keepInitial = False)
    jpdamot.targetset[i].reset2initialstate()

    xf0 = jpdamot.targetset[i].groundtruthrecorder.getvar_byidx('xfk',0)
    Pf0 = jpdamot.targetset[i].groundtruthrecorder.getvar_byidx('Pfk',0)

    xf0,_ = genRandomMeanCov(xf0,Pf0,1,np.eye(len(xf0)))
    jpdamot.targetset[i].setInitialdata(xf0, Pf0, currt0 = 0,currt=0)

#%% Run the filter
for t,tk,dt in simmanger.iteratetimesteps():
    print (t,tk,dt)
    Uk = None
    jpdamot.propagate(t, dt, Uk)
    # after prop, the time is t+dt

    # generate a random measurement
    Zkset = clc.defaultdict(list)

    targetIDlist = jpdamot.targetset.targetIDs()

    grounttruthDA= {}

    for i in range(jpdamot.targetset.ntargs):
        xk = jpdamot.targetset[i].groundtruthrecorder.getvar_bytime('xfk',t+dt)[0]
        ZZ = jpdamot.sensorset.generateRndMeasSet(t+dt,dt,xk)
        # ZZ = {'ID': {'zk': zk, 'isinsidek':isinsidek, 'Lk': Lk   } }
        for sensID in ZZ:
            Zkset[sensID].append( ZZ[sensID]['zk'] )
            if sensID not in grounttruthDA:
                grounttruthDA[sensID] = pd.DataFrame(index =targetIDlist )
            n=len(Zkset[sensID])-1
            grounttruthDA[sensID].at[jpdamot.targetset[i].ID,n]=1

    for sensID in jpdamot.sensorset.sensorIDs():
        if sensID in grounttruthDA:
            grounttruthDA[sensID].fillna(0,inplace=True)

    jpdamot.setgrounttruthDA( grounttruthDA )
    jpdamot.set_DAmat_from_groundtruthDA()

    # Zkset should be {'sensID1':[zk1,zk2], 'sensID2':[zk1,zk2,zk3],}
    jpdamot.measUpdt(t+dt,dt, Zkset)



fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(jpdamot.targetset.ntargs):
    xfk = jpdamot.targetset[i].recorderpost.getvar_alltimes('xfk')
    ax.plot(xfk[:,0],xfk[:,1])

plt.show()

#%% metrics
Errt=[None]*jpdamot.targetset.ntargs
Rmse=[None]*jpdamot.targetset.ntargs
for i in range(jpdamot.targetset.ntargs):
    xt = jpdamot.targetset[i].groundtruthrecorder.getvar_alltimes('xfk')
    xf = jpdamot.targetset[i].recorderpost.getvar_alltimes('xfk')

    errt,rmse = uqmetrics.getEsterror(xt,xf,
                                      stateset={'state':[0,1,2,3],
                                                'pos':[0,1],
                                                'vel':[2,3]
                                                })
    Errt[i] = errt
    Rmse[i] = rmse


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(simmanger.tvec,Errt[0]['state'])
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(simmanger.tvec,Errt[0]['pos'])
plt.pause(0.1)




simmanger.savefigure(fig,['post','K'],'test.png')
plt.show()
#%% Saving
simmanger.finalize()

simmanger.save(metalog,mainfile=__file__, Errt=Errt,Rmse=Rmse, jpdamot=jpdamot )

debugStatuslist = jpdamot.debugStatus()
uqutilhelp.DebugStatus().writestatus(debugStatuslist,simmanger.debugstatusfilepath)






