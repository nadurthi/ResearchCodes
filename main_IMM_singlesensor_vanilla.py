"""IMM single sensor code."""

import matplotlib
import matplotlib.pyplot as plt
import loggerconfig
import numpy as np
from scipy.linalg import block_diag
import collections as clc
import os
import numpy.linalg as npalg
from uq.filters.kalmanfilter import KFfilterer
from uq.filters import imm as immfilter
from uq.filters import gmmbase as uqfgmmbase
from physmodels import motionmodels as phymm
from physmodels.sensormodels import DiscLTSensorModel
from uq.uqutils import metrics as uqmetrics
from uq.uqutils import simmanager as uqsimmanager

matplotlib.use('TkAgg')
plt.close('all')

logger = loggerconfig.logging.getLogger(__name__)

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

simname = "IMM_test"

metalog = """
IMM test on tracking 1 target and 1 sensor

IMM integrated simmulator

Author: Venkat
"""


simmanger = uqsimmanager.SimManager(t0=0, tf=200, dt=2, dtplot=0.1,
                                    simname=simname,
                                    savepath="simulations",
                                    workdir=os.getcwd())

simmanger.initialize(repocheck=False)

# %%

modefilterer = KFfilterer()

H = np.hstack((np.eye(2), np.zeros((2, 3))))
R = block_diag((0.5)**2, (0.5)**2)

sensormodel = DiscLTSensorModel(H, R, recorderobj=None, recordSensorState=False)

um1 = phymm.KinematicModel_UM_5state()
ct2 = phymm.KinematicModel_CT()
p = np.array([[0.95, 0.05], [0.05, 0.95]])
dynMultiModels = phymm.MultipleMarkovMotionModel([um1, ct2], p)

vmag = 2
xfk = np.random.rand(5)
xfk[4] = 0.1
xfk[2:4] = xfk[2:4]*vmag/npalg.norm(xfk[2:4])

Pfk = np.random.randn(5,5)
Pfk = np.matmul(Pfk,Pfk.T)

gmm0 = uqfgmmbase.GMM.fromlist([xfk,xfk],[Pfk,Pfk],[0.5,0.5],0)
modelprob=np.array([0.5,0.5])

immf = immfilter.IntegratedIMM(dynMultiModels,0,gmm0,modelprob, sensormodel, modefilterer,
                 recorderobjprior=None, recorderobjpost=None,
                 recordfilterstate=True)

# % get ground truth
xk = xfk.copy()
immf.groundtruthrecorder.record(0,xfk=xk)
nt = simmanger.ntimesteps
for t,tk,dt in simmanger.iteratetimesteps():
    if 0<=tk<20:
        xk[4]=0
    if 20<=tk<30:
        xk[4]=0.2
    if 30<=tk<50:
        xk[4]=0
    if 50<=tk<60:
        xk[4]=-0.2
    if 60<=tk<80:
        xk[4]=0
    if 80<=tk<100:
        xk[4]=0.2

    _,xk = ct2.propforward( t, dt, xk, uk=0)
    immf.groundtruthrecorder.record(t+dt,xfk=xk)

xtruth = immf.groundtruthrecorder.getvar_alltimes_stacked('xfk')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xtruth[:,0],xtruth[:,1])

plt.show()

#%% Run the filter

for t,tk,dt in simmanger.iteratetimesteps():
    print (t,tk,dt)
    uk = None
    immf.propagate(t,dt,uk)
    # after prop, the time is t+dt


    # generate a random measurement
    xk = immf.groundtruthrecorder.getvar_bytime('xfk',t+dt)
    zk,isinsidek,Lk = immf.sensormodel.generateRndMeas(t+dt, dt, xk,useRecord=False)
    immf.sensormodel.recorder.record(t+dt,zk=zk)
    # Zkset should be {'sensID1':[zk1,zk2], 'sensID2':[zk1,zk2,zk3],}
    immf.measUpdate(t+dt,dt, zk)


xfk = immf.recorderpost.getvar_alltimes_stacked('xfk')
fig = plt.figure()
ax = fig.add_subplot(111)
truplot, = ax.plot(xtruth[:,0],xtruth[:,1],label = 'truth')
estplot, = ax.plot(xfk[:,0],xfk[:,1],label = 'est')
plt.legend(handles=[truplot, estplot])
plt.show()

modelprobfk = immf.recorderpost.getvar_alltimes_stacked('modelprobfk')

fig = plt.figure()
ax = fig.add_subplot(111)
truplot, = ax.plot(simmanger.tvec,xtruth[:,4],label = 'truth')
estplot, = ax.plot(simmanger.tvec,xfk[:,4],label = 'est')
plt.legend(handles=[truplot, estplot])
plt.show()

fig = plt.figure('gg')
ax = fig.add_subplot(111)
mp0, = ax.plot(simmanger.tvec,modelprobfk[:,0],label = '0')
mp1, = ax.plot(simmanger.tvec,modelprobfk[:,1],label = '1')
plt.legend(handles=[mp0, mp1])
plt.show()


#%% metrics

IMMmetrics = uqmetrics.Metrics()

IMMmetrics.Errt=clc.defaultdict(list)
IMMmetrics.Rmse=clc.defaultdict(list)

#%% Saving
simmanger.finalize()

simmanger.save(metalog,mainfile=__file__, immf=immf,IMMmetrics=IMMmetrics )

#debugStatuslist = jpdamot.debugStatus()
#uqutilhelp.DebugStatus().writestatus(debugStatuslist,simmanger.debugstatusfilepath)






