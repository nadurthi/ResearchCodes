# -*- coding: utf-8 -*-
import logging
import numpy as np
import numpy.linalg as nplg
import scipy.linalg as sclg
import uuid
import collections as clc
from scipy.linalg import block_diag
import pdb
import uq.stats.samplers as uqstsamp
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from uq.uqutils import recorder


        
class SensorModel:
    """
    The
    """
    sensorName = 'SensorModel'
    def __init__(self, recordSensorState=False,recorderobj=None,sensorrecorder=None, **kwargs):
        
        self.ID = uuid.uuid4()
        self.recordSensorState = recordSensorState

        self.sensStateStrs = [] #states-str of the sensor measurment ['r','th']
        self.sensstates = [] # states required to make the measurement as time time t
        self.currt = 0

        if recorderobj is None:
            self.recorder = recorder.StatesRecorder_list(statetypes = ['zk']+self.sensstates )
        else:
            self.recorder = recorderobj

    def getSenStateDict(self):
        return {}
    
    def updateparam(self, currt, **params):
        self.currt = currt

    def measNoise(self, t, dt, xk, **params):
        return self.R

    def H(self, t, dt, xk, **params):
        pass

    def recordHistory(self, z,t):
        raise NotImplementedError("ToDo")
        n = len(self.sensstates)
        SensorState = clc.namedtuple('SensorState', ['tk', 'zk']
                                     + self.sensstates)
        ss = []
        for i in range(n):
            ss.append(getattr(self, self.sensstates[i]))

        ss = [self.currt, z] + ss
        self.sensorHistory.append(SensorState(**ss))

    def __call__(self,t, dt, xk):
        raise NotImplementedError('implement the __Call__ here in subclass')

    def dynstate(self):
        ss={}
        for st in self.sensstates:
            ss['st'] = getattr(self,st,None)
        return ss

    def generateRndMeas(self, t, dt, xk,useRecord=False):
        R = self.measNoise(t, dt, xk)
        zk,isinsidek,Lk = self.__call__(t, dt, xk)
        # pdb.set_trace()
        zk = zk + np.matmul(sclg.sqrtm(R), np.random.randn(self.hn))

        if self.recordSensorState is True:
            self.recorder.record(t,zk=zk,**self.getSenStateDict())

        return zk,isinsidek,Lk
    
    def detection(self, t, dt, Xk):
        XXk = Xk.copy()
        ndim = XXk.ndim
        if ndim ==1:
            XXk=XXk.reshape(1,-1)

            
        Zk=np.zeros(XXk.shape[0])
        dets = [1,0]
        for i in range(XXk.shape[0]):
            xk = XXk[i]
            zk,isinsidek,Lk = self.__call__(t, dt, xk)
            if isinsidek:
                # inside so use TP and FN
                Zk[i]=1
            else: 
                # outside so use TN and FP
                Zk[i]=0
                

        if ndim ==1:
            Zk=Zk[0]
            
        return Zk,None,None
    
    def generateRndDetections(self, t, dt, Xk,useRecord=False):
        """
        return detection (1) or not (0) for each point in Xk
        TP: inside FOV and detected inside FOV
        FP: outside FOV and detected inside FOV
        TN: outside FOV and detected outside FOV
        FN: inside FOV and detected as outside FOV
        """
        XXk = Xk.copy()
        ndim = XXk.ndim
        if ndim ==1:
            XXk=XXk.reshape(1,-1)
            
            
        Zk=np.zeros(XXk.shape[0])
        dets = [1,0]
        for i in range(XXk.shape[0]):
            xk = XXk[i]
            zk,isinsidek,Lk = self.__call__(t, dt, xk)
            if isinsidek:
                # inside so use TP and FN
                idx = uqstsamp.samplePMF([self.TP,self.FN],1)
                Zk[i]=dets[idx[0]]
            else: 
                # outside so use TN and FP
                idx = uqstsamp.samplePMF([self.FP,self.TN],1)
                Zk[i]=dets[idx[0]]
                


        if self.recordSensorState is True:
            self.recorder.record(t,zk=zk,**self.getSenStateDict())
        if ndim ==1:
            Zk=Zk[0]
            
        return Zk,None,None
    
    def debugStatus(self):
        ss=[]
        ss.append(['----------','----------'])
        ss.append(['SensorModel',self.sensorName])
        ss.append(['----------','----------'])
        ss.append(['ID',self.ID])
        ss.append(['recordSensorState',self.recordSensorState])
        ss.append(['currt',self.currt])
        ss.append(['hn',self.hn])

        return ss


class SensorSet:
    """
    Measurement set is generated as
    ZZ={
            sensID1:{'zk':z1,'isinsidek':i1,'Lk':l1, }
            sensID2:{'zk':z1,'isinsidek':i1,'Lk':l1, }

        }
    only one measurement for each sensor
    """
    sensorName = 'SensorSet'
    def __init__(self):
        self.sensormodels = []
        self.ID = uuid.uuid4()
        self.ID2model = {}

    def debugStatus(self):
        ss=[]
        ss.append(['----------','----------'])
        ss.append(['SensorSet',self.ID])
        ss.append(['----------','----------'])
        ss.append(['nsensors',self.nsensors])
        for i in range(self.nsensors):
            ss = ss + [['\t'+s1[0],s1[1]] for s1 in self.sensormodels[i].debugStatus() ]

        return ss


    def sensorIDs(self):
        return [ss.ID for ss in self.sensormodels]

    @property
    def nsensors(self):
        return len(self.sensormodels)

    def getbyID(self,ID):
        return self.__getitem__( self.ID2model[ID] )

    def __getitem__(self,i):
        return self.sensormodels[i]


    def addSensor(self, sensormodel):
        self.sensormodels.append(sensormodel)

        for i in range(len(self.sensormodels)):
            self.ID2model[self.sensormodels[i].ID] = i

    def __call__(self,t, dt, xk,sensorIDs=None):
        if sensorIDs is None:
            sensorIDs = [x.ID for x in self.sensormodels]

        ZZ={ID:{'zk':[],'isinsidek':[],'Lk':[]} for ID in sensorIDs}

        for sensID in sensorIDs:
            senidx = self.ID2model[sensID]
            zk,isinsidek,Lk = self.sensormodels[senidx].__call__(t, dt, xk)
            ZZ[sensID]['zk'] = zk
            ZZ[sensID]['isinsidek'] = isinsidek
            ZZ[sensID]['Lk'] = Lk



        return ZZ


    def deleteSensor(self, sensorIDs):
        self.sensormodels = filter(lambda x: x.ID not in sensorIDs, self.sensormodels)

        for i in range(len(self.sensormodels)):
            self.ID2model[self.sensormodels[i].ID] = i

    def getStackedSensorSubset(self,sensorIDs):
        sensormodelsubset = [ss for ss in self.sensormodels if ss.ID in sensorIDs]
        return StackedSensorModel(sensormodelsubset)


    def generateRndMeasSet(self,t, dt, xk,sensorIDs=None):

        if sensorIDs is None:
            sensorIDs = [x.ID for x in self.sensormodels]

        ZZ={ID:{'zk':[],'isinsidek':[],'Lk':[]} for ID in sensorIDs}

        for sensID in sensorIDs:
            idx = self.ID2model[sensID]
            zk,isinsidek,Lk = self.sensormodels[idx].generateRndMeas(t, dt, xk)
            ZZ[sensID]['zk'] = zk
            ZZ[sensID]['isinsidek'] = isinsidek
            ZZ[sensID]['Lk'] = Lk


        return ZZ



    

class StackedSensorModel:
    sensorName = 'StackedSensorModel'
    def __init__(self,sensormodels,recordSensorState=False):
        self.sensormodels = sensormodels

        self.ID = uuid.uuid4()
        self.recordSensorState = recordSensorState
        self.sensorHistory = []
        self.sensstates = []
        self.currt = 0

        self.hn=np.sum([sensormodel.hn for sensormodel in self.sensormodels])

        for i in range(len(self.sensormodels)):
            self.sensstates.extend( self.sensormodels[i].sensstates  )

    def debugStatus(self):
        ss=[]
        ss.append(['----------','----------'])
        ss.append(['SensorModel:',self.sensorName])
        ss.append(['----------','----------'])
        ss.append(['ID',self.ID])
        ss.append(['recordSensorState',self.recordSensorState])
        ss.append(['currt',self.currt])

        ss.append(['hn',self.hn])

        return ss

    def measNoise(self, t, dt, xk, **params):
        R = []
        for i in range(len(self.sensormodels)):
            R.append(self.sensormodels[i].measNoise(t, dt, xk, **params))

        return block_diag(R)

    def __call__(self,t, dt, xk):
        raise NotImplementedError("ToDo")
        zk=[]
        isinsidek=[]
        Lk=[]
        for i in range(len(self.sensormodels)):
            siD = self.sensormodels[i].ID
            D = self.sensormodels[i].__call__(t, dt, xk)
            zk.append(D[siD][0])
            isinsidek.append(D[siD][1])
            Lk.append(D[siD][2])

        if zk.ndim ==1:
            zk = np.hstack(zk)

        try:
            zk = np.stack(zk,axis=0)
        except:
            print("all zks are not the same size in StackedSensorModel")

        isinsidek = np.hstack(isinsidek)
        Lk = np.hstack(Lk)

        return zk, isinsidek, Lk

    def updateparam(self, currt, sensidx,**params):
        self.currt = currt
        self.sensormodels[sensidx].updateparam( currt, **params)

    def generateRndMeas(self, t, dt, xk):
        raise NotImplementedError("ToDo")
        zk=[]
        for i in range(len(self.sensormodels)):
            siD = self.sensormodels[i].ID
            z = self.sensormodels[i].__call__(t, dt, xk)
            zk.append(z)

        zk = np.hstack(zk)

        R = self.measNoise(t, dt, xk)

        zk = zk + np.matmul(sclg.sqrtm(R), np.random.randn(self.hn))

        if self.recordSensorState is True:
            self.recordHistory(zk)

        return zk


    def recordHistory(self, z):
        for i in range(len(self.sensormodels)):
            self.sensormodels[i].recordHistory(None)

        n = len(self.sensstates)
        SensorState = clc.namedtuple('SensorState', ['tk', 'zk'] )

        self.sensorHistory.append(SensorState(self.currt,z))


class DiscLTSensorModel(SensorModel):
    sensorName = 'DiscLTSensorModel'
    def __init__(self, H, R, recordSensorState=False,**kwargs):
        self.Hmat = H
        self.R = R
        self.hn = self.Hmat.shape[0]
        self.sensStateStrs = ['x','y']
        self.sensstates = [] # states required to make the measurement as time time t

        super().__init__(recordSensorState=recordSensorState,**kwargs)

    def measNoise(self, t, dt, xk, **params):
        return self.R

    def H(self, t, dt, xk, **params):
        return self.Hmat

    def __call__(self,t, dt, xk):
        zk = np.matmul(self.Hmat, xk)
        return [zk, 1, 0]

class Disc2DRthetaSensorModel(SensorModel):
    sensorName = 'DiscRthetaSensorModel'
    def __init__(self, R, recordSensorState=False,**kwargs):
        self.R = R
        self.hn = 2
        self.sensStateStrs = ['r','th']
        self.sensstates = [] # states required to make the measurement as time time t

        super().__init__(recordSensorState=recordSensorState,**kwargs)

    def measNoise(self, t, dt, xk, **params):
        return self.R

    def H(self, t, dt, xk, **params):
        H=np.zeros((2,len(xk)))
        r = np.sqrt(xk[0]**2 + xk[1]**2)
        th = np.arctan2(xk[1],xk[0])
        H[0,0:2] = (1/r)*xk[:2]
        H[1,0:2] =  (1/r**2)*np.array([ -xk[1], xk[0] ])
                
        return H

    def __call__(self,t, dt, xk):
        r = np.sqrt(xk[0]**2 + xk[1]**2)
        th = np.arctan2(xk[1],xk[0])
        
        zk = np.array([r, th])
        isinFOV = 1
        L = 0
        return [zk, isinFOV, L]



