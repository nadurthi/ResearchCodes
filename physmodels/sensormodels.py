# -*- coding: utf-8 -*-
import logging
import numpy as np
import numpy.linalg as nplg
import scipy.linalg as sclg
import uuid
import collections as clc
from scipy.linalg import block_diag

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from uq.uqutils import recorder


class SensorModel:
    """
    The
    """
    sensorName = 'SensorModel'
    def __init__(self, recordSensorState=False,sensorrecorder=None, **kwargs):
        self.ID = uuid.uuid4()
        self.recordSensorState = recordSensorState
        
        self.sensstates = []
        self.currtk = 0
        
        if recorder is None:
            self.recorder = recorder.StatesRecorder_list(statetypes = {'zk':(None,),'t':(None,),'sensstates':(None,1)} )
        else:
            self.recorder = recorder
            
            
    def updateparam(self, updatetk, **params):
        if updatetk:
            self.currtk += 1

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

        ss = [self.currtk, z] + ss
        self.sensorHistory.append(SensorState(**ss))

    def __call__(self,t, dt, xk):
        raise NotImplementedError('implement the __Call__ here in subclass')

    def generateRndMeas(self, t, dt, xk):
        R = self.measNoise(t, dt, xk)
        zk,isinsidek,Lk = self.__call__(t, dt, xk)
        zk = zk + np.matmul(sclg.sqrtm(R), np.random.randn(self.hn))
        
        if self.recordSensorState is True:
            self.recordHistory(zk,t)

        return zk,isinsidek,Lk

    def debugStatus(self):
        ss=[]
        ss.append(['----------','----------'])
        ss.append(['SensorModel',self.sensorName])
        ss.append(['----------','----------'])
        ss.append(['ID',self.ID])
        ss.append(['recordSensorState',self.recordSensorState])
        ss.append(['currtk',self.currtk])
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
        self.currtk = 0
        
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
        ss.append(['currtk',self.currtk])
        
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
        
        return {self.ID: [zk, isinsidek, Lk]}
    
    def updateparam(self, updatetk, **params):
        if updatetk:
            self.currtk += 1
        if len(params) == 0:
            for i in range(len(self.sensormodels)):
                self.sensormodels[i].updateparam( updatetk)
        else:
            for i in range(len(self.sensormodels)):
                self.sensormodels[i].updateparam( updatetk, params[i])
                
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
        
        self.sensorHistory.append(SensorState(self.currtk,z))
        
    
class DiscLTSensorModel(SensorModel):
    sensorName = 'DiscLTSensorModel'
    def __init__(self, H, R, recordSensorState=False,**kwargs):
        self.Hmat = H
        self.R = R
        self.hn = self.Hmat.shape[0]
        self.sensstates = ['x','y']
        
        super().__init__(recordSensorState=False,**kwargs)

    def measNoise(self, t, dt, xk, **params):
        return self.R

    def H(self, t, dt, xk, **params):
        return self.Hmat

    def __call__(self,t, dt, xk):
        zk = np.matmul(self.Hmat, xk)
        return [zk, 1, 0]
        


class PlanarSensorModel(SensorModel):
    sensorName = 'PlanarSensorModel'
    def __init__(self, xc, R, posstates=None, enforceConstraint=False,
                 rmax=1, alpha=1, phi=1,recordSensorState=False, **kwargs):
        self.hn = None
        self.rmax = rmax  # FOV
        self.alpha = alpha  # FOV
        self.xc = xc
        self.R = R  # noise covariance
        self.phi = phi  # look direction
        self.posstates = posstates  # ideally the [0,1] of xk
        self.lRg = np.array([[np.cos(phi), -np.sin(phi)],
                             [np.sin(phi), np.cos(phi)]])
        self.ltg = xc
        self.enforceConstraint = enforceConstraint

        self.sensstates = ['lRg', 'ltg', 'rmax', 'alpha']

        super().__init__(recordSensorState=False,**kwargs)

    def updateparam(self, updatetk, **params):
        self.rmax = params.get('rmax', self.rmax)
        self.alpha = params.get('alpha', self.alpha)
        self.xc = params.get('xc', self.xc)
        self.R = params.get('R', self.R)
        self.phi = params.get('phi', self.phi)

        self.lRg = np.array([[np.cos(phi), -np.sin(phi)],
                             [np.sin(phi), np.cos(phi)]])
        self.ltg = xc

        super(PlanarSensorModel, self).updateparam(updatetk, **kwargs)


class RcircularFOVsensor(PlanarSensorModel):
    sensorName = 'RcircularFOVsensor'
    """
    [r]
    """

    def __init__(self, xc, R, posstates=None, enforceConstraint=False,
                 rmax=1, alpha=1, phi=1,recordSensorState=False, **kwargs):
        """
        enforceConstraint

        """
        self.hn = 1
        super().__init__(xc, R, posstates=None, enforceConstraint=False,
                         rmax=1, alpha=1, phi=1, recordSensorState=False,**kwargs)

    def penaltyFOV(z, angFOVdev, rFOVdev):
        L = 0
        isinFOV = 1
        if rFOVdev > 0 or np.abs(angFOVdev) > self.alpha:
            L = 100
            isinFOV = 0
        return isinFOV, L

    def __call__(self,t, dt, xk):
        """
        xk is [x,y,.....] when posstates is None
        """
        if self.posstates is None:
            xkp = xk[:2]
        else:
            xkp = xk[self.posstate]

        r = np.sqrt((xkp[0] - xc[0])**2 + (xkp[1] - xc[1])**2)
        th = np.atan2(xkp[1] - xc[1], xkp[0] - xc[0])

        angFOVdev = phi - th
        rFOVdev = r - self.rmax

        z = r
        isinFOV = 1
        L = 0
        if self.enforceConstraint:
            isinFOV, L = self.penaltyFOV(z, angFOVdev, rFOVdev)

        return [z, isinFOV, L]


class THcircularFOVsensor(PlanarSensorModel):
    sensorName = 'THcircularFOVsensor'
    """
    [th]
    """

    def __init__(self, xc, R, posstates=None, enforceConstraint=False,
                 rmax=1, alpha=1, phi=1,recordSensorState=False, **kwargs):
        """
        enforceConstraint

        """
        self.hn = 1
        super().__init__(xc, R, posstates=None, enforceConstraint=False,
                         rmax=1, alpha=1, phi=1,recordSensorState=False, **kwargs)

    def penaltyFOV(z, angFOVdev, rFOVdev):
        L = 0
        isinFOV = 1
        if rFOVdev > 0 or np.abs(angFOVdev) > self.alpha:
            L = 100
            isinFOV = 0
        return isinFOV, L

    def __call__(self,t, dt, xk):
        """
        xk is [x,y,.....] when posstates is None
        """
        if self.posstates is None:
            xkp = xk[:2]
        else:
            xkp = xk[self.posstate]

        r = np.sqrt((xkp[0] - xc[0])**2 + (xkp[1] - xc[1])**2)
        th = np.atan2(xkp[1] - xc[1], xkp[0] - xc[0])

        angFOVdev = phi - th
        rFOVdev = r - self.rmax

        z = th
        isinFOV = 1
        L = 0
        if self.enforceConstraint:
            isinFOV, L = self.penaltyFOV(z, angFOVdev, rFOVdev)
        
        return [z, isinFOV, L]
        


class RTHcircularFOVsensor(PlanarSensorModel):
    sensorName = 'RTHcircularFOVsensor'
    """
    [r,th]
    """

    def __init__(self, xc, R, posstates=None, enforceConstraint=False,
                 rmax=1, alpha=1, phi=1,recordSensorState=False, **kwargs):
        """
        enforceConstraint

        """
        self.hn = 2
        super().__init__(xc, R, posstates=None, enforceConstraint=False,
                         rmax=1, alpha=1, phi=1,recordSensorState=False, **kwargs)

    def penaltyFOV(z, angFOVdev, rFOVdev):
        L = 0
        isinFOV = 1
        if rFOVdev > 0 or np.abs(angFOVdev) > self.alpha:
            L = 100
            isinFOV = 0
        return isinFOV, L

    def __call__(self,t, dt, xk):
        """
        xk is [x,y,.....] when posstates is None
        """
        if self.posstates is None:
            xkp = xk[:2]
        else:
            xkp = xk[self.posstate]

        r = np.sqrt((xkp[0] - xc[0])**2 + (xkp[1] - xc[1])**2)
        th = np.atan2(xkp[1] - xc[1], xkp[0] - xc[0])

        angFOVdev = phi - th
        rFOVdev = r - self.rmax

        z = np.array([r, th])
        isinFOV = 1
        L = 0
        if self.enforceConstraint:
            isinFOV, L = self.penaltyFOV(z, angFOVdev, rFOVdev)
        
        return [z, isinFOV, L]
        


