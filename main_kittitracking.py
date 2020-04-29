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
from uq.motfilter.jpda import JPDAMOT 
from uq.filters.kalmanfilter import TargetKF
from uq.motfilter.targets import Target
from uq.uqutils.random import genRandomMeanCov
from physmodels.motionmodels import KinematicModel_UM
from physmodels.sensormodels import DiscLTSensorModel
import uq.motfilter.measurements as motmeas
from uq.uqutils import recorder as uqrecorder
from uq.uqutils import helper as uqutilhelp

#%%
mu = np.zeros(3)
P = np.eye(3)
X, w = uqcb.UT_sigmapoints(mu, P)

X, w = uqcb.CUT4pts_gaussian(mu, P)


#%%

filterer = TargetKF()

H = np.hstack((np.eye(2),np.zeros((2,2))))
R = block_diag((1e-1)**2,(1e-1)**2)

sensormodel = DiscLTSensorModel(H,R,recordSensorState=False)

jpdamot = JPDAMOT(filterer,recordMeasurementSets=True)
jpdamot.sensorset.addSensor(sensormodel)


simtime = uqutilhelp.SimTime(t0=0,tf=55,dt=0.5,dtplot=0.1)


vmag = 2

# adding targets
for i in range(5):
    xfk = np.random.rand(4)
    xfk[0:2] = 10*xfk[0:2]
    xfk[2:4] = vmag*xfk[2:4]
    Pfk = np.random.randn(4,4)
    Pfk = np.matmul(Pfk,Pfk.T)
    
    dynmodel = KinematicModel_UM()
    
    recorder = uqrecorder.StatesRecorder_fixedDim(statetypes = {'t':(1,),'xfk':(1,dynmodel.fn,),'Pfk':(1,dynmodel.fn,dynmodel.fn)})
    target = Target(dynModels=[dynmodel], xfk=[xfk], Pfk=[Pfk], currtk = 0, recordfilterstate=True,
            status='active', recorder = recorder,filterer=filterer,saveinitializingdata=True)
    
    jpdamot.targetset.addTarget(target)
    
    
            


#%% get ground truth
for t,tk,dt in simtime.iteratetime():           
    print (t,tk,dt)
    Uk = None
    jpdamot.propagate(t, dt, Uk)  

# jpdamot.filterer.debugStatus()
debugStatuslist = jpdamot.debugStatus()
uqutilhelp.DebugStatus().writestatus(debugStatuslist,'debugtest.txt')

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(jpdamot.targetset.ntargs):
    xfk = jpdamot.targetset[i].recorder.getvar_alltimes('xfk')
    ax.plot(xfk[:,0,0],xfk[:,0,1])

plt.show()
#%% Save ground truth and reset filters to new intiial conditions
for i in range(jpdamot.targetset.ntargs):
    jpdamot.targetset[i].groundtruthrecorder=  jpdamot.targetset[i].recorder.makecopy()
    jpdamot.targetset[i].recorder.cleardata(keepInitial = False)
    jpdamot.targetset[i].reset2initialstate()
    
    xf0 = jpdamot.targetset[i].groundtruthrecorder.getvar_byidx('xfk',0)[0]
    Pf0 = jpdamot.targetset[i].groundtruthrecorder.getvar_byidx('Pfk',0)[0]
    
    xf0,_ = genRandomMeanCov(xf0,Pf0,1,np.eye(len(xf0)))
    jpdamot.targetset[i].setInitialdata([xf0], [Pf0], currt0 = 0,currt=0)
    
#%% Run the filter
for t,tk,dt in simtime.iteratetime():           
    print (t,tk,dt)
    Uk = None
    jpdamot.propagate(t, dt, Uk)  
    
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
    jpdamot.measUpdt(t,dt, Zkset)
    
    
    
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(jpdamot.targetset.ntargs):
    xfk = jpdamot.targetset[i].recorder.getvar_alltimes('xfk')
    ax.plot(xfk[:,0,0],xfk[:,0,1])

plt.show()    
#%% Saving

