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


class SensorModel:
    def __init__(self, recordSensorState=False, **kwargs):
        self.ID = uuid.uuid4()
        self.recordSensorState = recordSensorState
        self.sensorHistory = []
        self.sensstates = []
        self.currtk = 0

    def updateparam(self, updatetk, **params):
        if updatetk:
            self.currtk += 1

    def measNoise(self, t, dt, xk, **params):
        return self.R

    def H(self, t, dt, xk, **params):
        pass

    def recordHistory(self, z):
        n = len(self.sensstates)
        SensorState = clc.namedtuple('SensorState', ['tk', 'zk'] 
                                     + self.sensstates)
        ss = []
        for i in range(n):
            ss.append(getattr(self, self.sensstates[i]))

        ss = [self.currtk, z] + ss
        self.sensorHistory.append(SensorState(**ss))

    def __call__(self):
        pass

    def generateRndMeas(self, t, dt, xk):
        R = self.measNoise(t, dt, xk)
        zk = self.__call__(t, dt, xk) + 
                np.matmul(sclg.sqrtm(R), np.random.randn(self.hn))
        
        if self.recordSensorState is True:
            self.recordHistory(zk)

        return zk

class StackedSensorModel:
    def __init__(self,sensormodels):
        self.sensormodels = sensormodels
        
        self.ID = uuid.uuid4()
        self.recordSensorState = recordSensorState
        self.sensorHistory = []
        self.sensstates = []
        self.currtk = 0
        
        self.hn=np.sum([sensormodel.hn for sensormodel in self.sensormodels])
        
        for i in range(len(self.sensormodels)):
            self.sensstates.extend( self.sensormodels[i].sensstates  )    
            
    def measNoise(self, t, dt, xk, **params):
        R = []
        for i in range(len(self.sensormodels)):
            R.append(self.sensormodels[i].measNoise(t, dt, xk, **params))
        
        return block_diag(R)
    
    def __call__(t, dt, xk):
        zk=[]
        isinsidek=[]
        Lk=[]
        for i in range(len(self.sensormodels)):
            z,isinside,L = self.sensormodels[i]__call__(t, dt, xk) 
            zk.append(z)
            isinsidek.append(isinside)
            Lk.append(Lk)
            
        zk = np.hstack(zk)
        isinsidek = np.hstack(isinsidek)
        Lk = np.hstack(Lk)
        
        return (zk, isinsidek, Lk)
    
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
        zk=[]
        for i in range(len(self.sensormodels)):
            z = self.sensormodels[i]__call__(t, dt, xk) 
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
    def __init__(self, H, R, **kwargs):
        self.H = H
        self.R = R
        self.hn = H.shape[0]
        self.sensstates = []
        super().__init__(**kwargs)

    def measNoise(self, t, dt, xk, **params):
        return self.R

    def H(self, t, dt, xk, **params):
        return self.H

    def __call__(t, dt, xk):
        zk = np.matmul(self.H, xk)
        return (z, 1, 0)


class PlanarSensorModel(SensorModel):
    def __init__(self, xc, R, posstates=None, enforceConstraint=False,
                 rmax=1, alpha=1, phi=1, **kwargs):
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

        super().__init__(**kwargs)

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
    """
    [r]
    """

    def __init__(self, xc, R, posstates=None, enforceConstraint=False,
                 rmax=1, alpha=1, phi=1, **kwargs):
        """
        enforceConstraint

        """
        self.hn = 1
        super().__init__(xc, R, posstates=None, enforceConstraint=False,
                         rmax=1, alpha=1, phi=1, **kwargs)

    def penaltyFOV(z, angFOVdev, rFOVdev):
        L = 0
        isinFOV = 1
        if rFOVdev > 0 or np.abs(angFOVdev) > self.alpha:
            L = 100
            isinFOV = 0
        return isinFOV, L

    def __call__(t, dt, xk):
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

        return (z, isinFOV, L)


class THcircularFOVsensor(PlanarSensorModel):
    """
    [th]
    """

    def __init__(self, xc, R, posstates=None, enforceConstraint=False,
                 rmax=1, alpha=1, phi=1, **kwargs):
        """
        enforceConstraint

        """
        self.hn = 1
        super().__init__(xc, R, posstates=None, enforceConstraint=False,
                         rmax=1, alpha=1, phi=1, **kwargs)

    def penaltyFOV(z, angFOVdev, rFOVdev):
        L = 0
        isinFOV = 1
        if rFOVdev > 0 or np.abs(angFOVdev) > self.alpha:
            L = 100
            isinFOV = 0
        return isinFOV, L

    def __call__(t, dt, xk):
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

        return (z, isinFOV, L)


class RTHcircularFOVsensor(PlanarSensorModel):
    """
    [r,th]
    """

    def __init__(self, xc, R, posstates=None, enforceConstraint=False,
                 rmax=1, alpha=1, phi=1, **kwargs):
        """
        enforceConstraint

        """
        self.hn = 2
        super().__init__(xc, R, posstates=None, enforceConstraint=False,
                         rmax=1, alpha=1, phi=1, **kwargs)

    def penaltyFOV(z, angFOVdev, rFOVdev):
        L = 0
        isinFOV = 1
        if rFOVdev > 0 or np.abs(angFOVdev) > self.alpha:
            L = 100
            isinFOV = 0
        return isinFOV, L

    def __call__(t, dt, xk):
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

        return (z, isinFOV, L)


