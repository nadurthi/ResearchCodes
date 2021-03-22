"""IMM+JPDA main function for testinf and simulation."""

import matplotlib
import numpy as np
from scipy.linalg import block_diag
import collections as clc
import os
import pandas as pd
from uq.motfilter import jpda
import uq.filters.kalmanfilter as uqkf
from uq.filters import sigmafilter as uqsigf
from uq.uqutils.random import genRandomMeanCov
from physmodels import motionmodels as phymm
from physmodels import sensormodels as physm
from physmodels import targets as phytarg
from uq.filters import imm as immfilter
from uq.gmm import gmmbase as uqfgmmbase
from uq.uqutils import recorder as uqrecorder
from uq.uqutils import metrics as uqmetrics
from uq.uqutils import simmanager as uqsimmanager
from uq.quadratures import cubatures as quadcub
import uq.quadratures.cubatures as uqcb
from uq.filters import mmpf as uqmmpf
from uq.filters import imm as uqimm

try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from loggerconfig import *


plt.close('all')
runfilename = __file__

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
ACC 2021 MMPF paper simulations
Author: Venkat
Date: June 4 2020

Comparing the IMM+JPDA filter using EKF and UT and CUT points
"""


simmanger = uqsimmanager.SimManager(t0=0,tf=55,dt=1,dtplot=0.1,
                                  simname="ACC-2021-MMPF",savepath="simulations",
                                  workdir=os.getcwd())

simmanger.initialize()




#%%



# H = np.hstack((np.eye(2),np.zeros((2,2))))
# R = block_diag((0.5)**2,(0.5)**2)
# sensormodel4d = physm.DiscLTSensorModel(H,R,recorderobj=None,recordSensorState=False)

H = np.hstack((np.eye(2), np.zeros((2, 3))))
# R = block_diag((0.5)**2, (0.5)**2)
# sensormodel5d = physm.DiscLTSensorModel(H, R, recorderobj=None, recordSensorState=False)

R = block_diag((5)**2, (1*np.pi/180)**2)
sensormodel5d = physm.Disc2DRthetaSensorModel(R, recorderobj=None, recordSensorState=False)

# sensorset = physm.SensorSet()
# sensorset.addSensor(sensormodel5d)
mmlist = {'EKFIMM':{},'UKFIMM':{},'CUT8IMM':{} ,'EKFMMPF':{}  ,'UKFMMPF':{},'CUT8MMPF':{}}
# mmlist = {'UKFIMM':{}  ,'UKFMMPF':{}}



modefilterer = uqkf.EKFfilterer()
immf = uqimm.TargetIMM(modefilterer)
mmlist['EKFIMM']['TargetFilter']=immf
mmlist['EKFIMM']['TargetSet']=phytarg.TargetSet()


modefilterer = uqsigf.Sigmafilterer(sigmamethod = quadcub.SigmaMethod.UT)
immf = uqimm.TargetIMM(modefilterer)
mmlist['UKFIMM']['TargetFilter']=immf
mmlist['UKFIMM']['TargetSet']=phytarg.TargetSet()

modefilterer = uqsigf.Sigmafilterer(sigmamethod = quadcub.SigmaMethod.CUT8)
immf = uqimm.TargetIMM(modefilterer)
mmlist['CUT8IMM']['TargetFilter']=immf
mmlist['CUT8IMM']['TargetSet']=phytarg.TargetSet()


modefilterer = uqkf.EKFfilterer()
mmpf = uqmmpf.TargetMMPF(modefilterer)
mmlist['EKFMMPF']['TargetFilter'] = mmpf
mmlist['EKFMMPF']['TargetSet']=phytarg.TargetSet()

modefilterer = uqsigf.Sigmafilterer(sigmamethod = quadcub.SigmaMethod.UT)
mmpf = uqmmpf.TargetMMPF(modefilterer)
mmlist['UKFMMPF']['TargetFilter'] = mmpf
mmlist['UKFMMPF']['TargetSet']=phytarg.TargetSet()

modefilterer = uqsigf.Sigmafilterer(sigmamethod = quadcub.SigmaMethod.CUT8)
mmpf = uqmmpf.TargetMMPF(modefilterer)
mmlist['CUT8MMPF']['TargetFilter'] = mmpf
mmlist['CUT8MMPF']['TargetSet']=phytarg.TargetSet()


MethodOrder = ['EKFIMM','UKFIMM','CUT8IMM','EKFMMPF'  ,'UKFMMPF','CUT8MMPF']

groundTruthTargets = phytarg.TargetSet()

vmag = 20

Npf = 500

# adding targets
for i in range(5):

    xfk = np.random.rand(4)
    xfk[0:2] = 1000*xfk[0:2]
    xfk[2:4] = vmag*xfk[2:4]
    Pfk = np.random.randn(4,4)
    Pfk = np.matmul(Pfk,Pfk.T)
    xfk5 = np.hstack([xfk,np.random.rand()])
    Pfk5 = block_diag(Pfk,0.1**2)
            
    dynmodel = phymm.KinematicModel_UM_5state()
    
    um1 = phymm.KinematicModel_UM_5state()
    ct2 = phymm.KinematicModel_CT()
    p=np.array([[0.95,0.05],[0.05,0.95]])
    dynMultiModels = phymm.MultipleMarkovMotionModel([um1,ct2],p)
    
    target = phytarg.Target(dynModel=dynmodel, xfk=xfk5, Pfk=Pfk5, currt = 0, recordfilterstate=True,
             recorderobjprior = None,recorderobjpost=None)
    target.groundtruthrecorder = uqrecorder.StatesRecorder_fixedDim(statetypes = {
            'xfk':(dynmodel.fn,)})
        
    groundTruthTargets.addTarget(target)
    
    for jn in mmlist.keys():
        if 'IMM' in jn:
            
            gmm0 = uqfgmmbase.GMM.fromlist([xfk5,xfk5],[Pfk5,Pfk5],[0.5,0.5],0)
            modelprob0=np.array([0.5,0.5])
            recorderobjprior = uqrecorder.StatesRecorder_list(statetypes = ['xfk','Pfk','gmmfk','modelprobfk'] )
            recorderobjpost = uqrecorder.StatesRecorder_list(statetypes = ['xfk','Pfk','gmmfk','modelprobfk'] )
            target = phytarg.Target(dynModelset=dynMultiModels, gmmfk=gmm0, modelprobfk = modelprob0, currt = 0, recordfilterstate=True,
                     recorderobjprior = recorderobjprior,recorderobjpost=recorderobjpost)

            target.groundtruthrecorder = uqrecorder.StatesRecorder_fixedDim(statetypes = {
                    'xfk':(ct2.fn,)})
            
        elif 'MMPF' in jn:
            recorderobjprior = uqrecorder.StatesRecorder_list(statetypes = ['xfk','Pfk','particlesfk'] )
            recorderobjpost = uqrecorder.StatesRecorder_list(statetypes = ['xfk','Pfk','particlesfk'] )
            target = phytarg.Target(dynModelset=dynMultiModels, currt = 0, recordfilterstate=True,
                     recorderobjprior = recorderobjprior,recorderobjpost=recorderobjpost)
    
            target.groundtruthrecorder = uqrecorder.StatesRecorder_fixedDim(statetypes = {
                    'xfk':(ct2.fn,)})
            


        target.freeze(recorderobj=False)
        mmlist[jn]['TargetSet'].addTarget(target)





#%% get ground truth

for n in range(groundTruthTargets.ntargs):
    xk = groundTruthTargets[n].xfk.copy()
    groundTruthTargets[n].groundtruthrecorder.record(0,xfk=xk)
    ss = np.sign(np.random.randn(1)[0])
    ss=-1
    nt = simmanger.ntimesteps
    for t,tk,dt in simmanger.iteratetimesteps():
        if 0<=tk<20:
            xk[4]=0
        if 20<=tk<40:
            xk[4]=0.3*ss
        if 40<=tk<60:
            xk[4]=0
        if 60<=tk<80:
            xk[4]=-0.3*ss
        if 80<=tk<100:
            xk[4]=0
        if 100<=tk<110:
            xk[4]=0.3*ss

        _,xk = ct2.propforward( t, dt, xk, uk=0)
        groundTruthTargets[n].groundtruthrecorder.record(t+dt,xfk=xk)



fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(groundTruthTargets.ntargs):
    xfk = groundTruthTargets[i].groundtruthrecorder.getvar_alltimes('xfk')
    ax.plot(xfk[:,0],xfk[:,1],label='Target: '+str(i))
    ax.plot(xfk[0,0],xfk[0,1],'x')
    ax.plot(xfk[-1,0],xfk[-1,1],'o')
    
ax.legend()
plt.show()
plt.pause(0.1)
simmanger.savefigure(fig, [], 'GroundTruth.png')

#%% Save ground truth and reset filters to new intiial conditions
modelprob0 = [0.5,0.5]
for i in range(groundTruthTargets.ntargs):

    xf0m = groundTruthTargets[i].groundtruthrecorder.getvar_byidx('xfk',0)
#    Pf0 = mmlist['IMMKFJPDA'].targetset[i].groundtruthrecorder.getvar_byidx('Pfk',0)
    Pf0 = groundTruthTargets[i].Pfk.copy()
    xf0,_ = genRandomMeanCov(xf0m,Pf0,0.8,np.eye(len(xf0m)))

    for jn in mmlist.keys():
        if 'IMM' in jn:
            gmm0 = uqfgmmbase.GMM.fromlist([xf0,xf0],[Pf0,Pf0],[0.5,0.5],0)
            modelprobf0 = np.array([0.5,0.5])
            m,P=gmm0.weightedest(modelprob0)
            mmlist[jn]['TargetSet'][i].setInitialdata(0,gmmfk=gmm0, modelprobfk=modelprobf0,xfk=m,Pfk=P)
            
        elif 'MMPF' in jn:
            r0samples= np.random.choice(len(modelprob0),size=Npf, replace=True, p=modelprob0)
            particlesfk = uqmmpf.MMParticleKF()
            for pp in range(Npf):
                xf0,_ = genRandomMeanCov(xf0m,Pf0,0.8,np.eye(len(xf0)))
                X={'xfk':xf0,'Pfk':Pf0,'r':r0samples[pp]}
                particlesfk.addParticle(X,1.0/Npf)
                
            mmlist[jn]['TargetSet'][i].setInitialdata(0,particlesfk=particlesfk,xfk=xf0,Pfk=Pf0)
        


#%% Run the filter
for t,tk,dt in simmanger.iteratetimesteps():
    print (t,tk,dt)
    Uk = None
    for jn in mmlist.keys():
        for i in range(groundTruthTargets.ntargs):
            mmlist[jn]['TargetFilter'].propagate(t, dt, mmlist[jn]['TargetSet'][i], Uk)
    # after prop, the time is t+dt
    
    Zk=[]
    for i in range(groundTruthTargets.ntargs):
        xk = groundTruthTargets[i].groundtruthrecorder.getvar_bytime('xfk',t+dt)[0]
        zk,_,_ = sensormodel5d.generateRndMeas(t+dt, dt, xk,useRecord=False)
        Zk.append(zk)
        

    for jn in mmlist.keys():
        for i in range(groundTruthTargets.ntargs):
            mmlist[jn]['TargetFilter'].measUpdate(t+dt, dt, mmlist[jn]['TargetSet'][i], sensormodel5d, Zk[i])

    pass

markers = ['b','g','m','c','r','y']
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(groundTruthTargets.ntargs):
    xfk = groundTruthTargets[i].groundtruthrecorder.getvar_alltimes('xfk')
    ax.plot(xfk[:,0],xfk[:,1],'k--')
    for idx,jn in enumerate(mmlist.keys()):
        xfk = mmlist[jn]['TargetSet'][i].recorderpost.getvar_alltimes('xfk')
        ax.plot(xfk[:,0],xfk[:,1],markers[idx], label=jn)


ax.legend()
plt.show()
plt.pause(0.1)
simmanger.savefigure(fig, [], 'Esttraj.png')

# %% metrics

mmmetrics = uqmetrics.Metrics()

mmmetrics.Errt = clc.defaultdict(list)
mmmetrics.Rmse = clc.defaultdict(list)

mmmetrics.MethodOrder = MethodOrder

xtlist = []
for j in range(groundTruthTargets.ntargs):
    xt = groundTruthTargets[j].groundtruthrecorder.getvar_alltimes('xfk') 
    xtlist.append(xt)
    
for jn in mmlist.keys():
    
    for i in range(groundTruthTargets.ntargs): # select the target
        xf = mmlist[jn]['TargetSet'][i].recorderpost.getvar_alltimes('xfk')
        xt = groundTruthTargets[i].groundtruthrecorder.getvar_alltimes('xfk')           
        # errt, rmse = uqmetrics.getEsterror2ClosestTarget(xtlist,xf,stateset={
        #                                       'state': [0, 1, 2, 3],
        #                                       'pos': [0, 1],
        #                                       'vel': [2, 3]
        #                                             })
        errt, rmse = uqmetrics.getEsterror(xt, xf, stateset={
                                              'pos': [0, 1],
                                              'vel': [2, 3]
                                                    })

        mmmetrics.Errt[jn].append(errt)
        mmmetrics.Rmse[jn].append(rmse)

df = pd.DataFrame()
cnt=0
for jn in mmlist.keys():
    for targidx, dd in enumerate(mmmetrics.Rmse[jn]): 
        df.loc[cnt,'Method'] = jn
        df.loc[cnt,'Target'] = targidx
        for key,value in dd.items():
            df.loc[cnt,key] = value
        cnt += 1

mmmetrics.dfrmse = df

dfpt = pd.pivot_table(df, values='pos', index=['Target'],
                    columns=['Method'], aggfunc=np.sum)
mmmetrics.dfrmse_state = dfpt

print(dfpt)

fig = plt.figure()
ax = fig.add_subplot(111)
for idx,jn in enumerate(mmlist.keys()):
    ax.plot(simmanger.tvec, mmmetrics.Errt[jn][0]['pos'], label=jn)

ax.legend()
plt.show()
plt.pause(0.1)
simmanger.savefigure(fig, ['post', 'K'], 'state.png')

fig = plt.figure()
ax = fig.add_subplot(111)
for idx,jn in enumerate(mmlist.keys()):
    ax.plot(simmanger.tvec, mmmetrics.Errt[jn][0]['vel'], label=jn)
    
plt.pause(0.1)
simmanger.savefigure(fig, ['post', 'K'], 'pos.png')

plt.show()
# %% IMM weights plots

mpfk = mmlist['UKFIMM']['TargetSet'][0].recorderpost.getvar_alltimes('modelprobfk')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(simmanger.tvec, mpfk[:,0], label='UM')
ax.plot(simmanger.tvec, mpfk[:,1], label='CT')
ax.legend()
plt.pause(0.1)
simmanger.savefigure(fig, ['post'], 'UKFIMM'+'immwts.png')

# %%
particlefk=mmlist['UKFMMPF']['TargetSet'][0].recorderpost.getvar_alltimes('particlesfk')

rt=[]
for i in range(len(particlefk)):
    RR=[]
    for j in particlefk[i].itersamplesIdx():
        RR.append(particlefk[i].X[j]['r'])
    CR=clc.Counter(RR)
    rt.append([CR[0],CR[1]])
    # print(i,"0: ",CR[0]," 1: ",CR[1])
rt=np.array(rt).astype(float)
s=np.sum(rt,axis=1)
rt[:,0]=rt[:,0]/s
rt[:,1]=rt[:,1]/s
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(simmanger.tvec, rt[:,0], label='UM')
ax.plot(simmanger.tvec, rt[:,1], label='CT')
ax.legend()
plt.pause(0.1)
simmanger.savefigure(fig, ['post'], 'UKFMMPF'+'immwts.png')

# %% Saving
simmanger.finalize()

simmanger.save(metalog, mainfile=runfilename, mmmetrics=mmmetrics, mmlist=mmlist)

simmanger.summarize()
# debugStatuslist = jpdamot.debugStatus()
# uqutilhelp.DebugStatus().writestatus(debugStatuslist,simmanger.debugstatusfilepath)
