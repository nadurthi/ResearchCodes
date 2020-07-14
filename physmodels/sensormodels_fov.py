import numpy as np
import numpy.linalg as nplag
from physmodels.sensormodels import SensorModel
import utils.plotting.geometryshapes as utpltgeom
from uq.uqutils import recorder

class PlanarSensorModel(SensorModel):
    sensorName = 'PlanarSensorModel'
    def __init__(self, xc=np.array([0,0]), R=np.eye(2), posstates=np.array([0,1]), 
                 TP=0.4, TN=0.4, FP=0.1, FN=0.1,
                 enforceConstraint=False, FOVradius=10, FOVcolor = 'r', 
                 FOValpha = 0.4 ,FOVsectorhalfangle=np.pi, dirn=0,
                 recordSensorState=False, **kwargs):
        super().__init__(recordSensorState=recordSensorState,**kwargs)
        
        self.hn = None
        self.FOVradius = FOVradius  # FOV
        self.FOVsectorhalfangle = FOVsectorhalfangle  # FOV
        self.xc = xc
        self.R = R  # noise covariance
        self.dirn = dirn  # look direction
        self.posstates = posstates  # ideally the [0,1] of xk
        self.FOVcolor = FOVcolor
        self.FOValpha = FOValpha
        self.TP = TP
        self.TN = TN
        self.FP = FP
        self.FN = FN
        
        self.enforceConstraint = enforceConstraint

        self.sensStateStrs = ['x','y']
        self.sensstates = ['R', 'xc', 'dirn', 'FOVradius', 'FOVsectorhalfangle'] # states required to make the measurement as time time t

        
        self.recorder = recorder.StatesRecorder_list(statetypes = ['zk']+self.sensstates )
    
    def getSenStateDict(self):
        D= {}
        for ss in self.sensstates:
            D[ss] = getattr(self,ss,None)
        
        return D
    
    def sensorPose(self):
        lRg = np.array([[np.cos(self.dirn), -np.sin(self.dirn)],
                             [np.sin(self.dirn), np.cos(self.dirn)]])
        ltg = self.xc
        
        return lRg, ltg
        
    def updateparam(self, currt, **params):
        self.FOVradius = params.get('FOVradius', self.FOVradius)
        self.FOVsectorhalfangle = params.get('FOVsectorhalfangle', self.FOVsectorhalfangle)
        self.xc = params.get('xc', self.xc)
        self.R = params.get('R', self.R)
        self.dirn = params.get('dirn', self.dirn)

        self.lRg = np.array([[np.cos(self.dirn), -np.sin(self.dirn)],
                             [np.sin(self.dirn), np.cos(self.dirn)]])
        self.ltg = self.xc

        super().updateparam(currt, **params)

class RTHcircularFOVsensor(PlanarSensorModel):
    sensorName = 'RTHcircularFOVsensor'
    """
    [r,th]
    """

    def __init__(self, xc=np.array([0,0]), R=np.eye(2), posstates=None, enforceConstraint=False,
                 FOVradius=1, FOVsectorhalfangle=np.pi, dirn=1,recordSensorState=False, **kwargs):
        """
        enforceConstraint

        """
        
        super().__init__(xc, R, posstates=posstates, enforceConstraint=enforceConstraint,
                         FOVradius=FOVradius, FOVsectorhalfangle=FOVsectorhalfangle, 
                         dirn=dirn,recordSensorState=recordSensorState, **kwargs)
        self.sensStateStrs = ['r','th']
        self.hn = 2
    def plotsensorFOV(self,ax):
        Xfov = utpltgeom.getCirclePoints2D(self.xc[0:2],self.FOVradius,N=100)
        ax.fill(Xfov[:,0],Xfov[:,1],self.FOVcolor,alpha=self.FOValpha,
                edgecolor=self.FOVcolor,facecolor=self.FOVcolor)
        
    def penaltyFOV(self,z, angFOVdev, rFOVdev):
        L = 0
        isinFOV = 1
        if rFOVdev > 0: # or np.abs(angFOVdev) > self.FOVsectorhalfangle:
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

        r = np.sqrt((xkp[0] - self.xc[0])**2 + (xkp[1] - self.xc[1])**2)
        th = np.arctan2(xkp[1] - self.xc[1], xkp[0] - self.xc[0])

        angFOVdev = self.dirn - th
        rFOVdev = r - self.FOVradius

        z = np.array([r, th])
        isinFOV = 1
        L = 0
        if self.enforceConstraint:
            isinFOV, L = self.penaltyFOV(z, angFOVdev, rFOVdev)

        return [z, isinFOV, L]
    
    def evalBatchNoFOV(self,t, dt, Xk,posstates=[0,1]):
            
        d = Xk[:,posstates]-self.xc
        r = nplag.norm(d,axis=1)
        th = np.arctan2(d[:,1],d[:,0])
        Zk = np.vstack([r,th]).T
        
        return Zk
    
class RcircularFOVsensor(PlanarSensorModel):
    sensorName = 'RcircularFOVsensor'
    """
    [r]
    """

    def __init__(self, xc, R, posstates=None, enforceConstraint=False,
                 FOVradius=1, FOVsectorhalfangle=1, dirn=1,recordSensorState=False, **kwargs):
        """
        enforceConstraint

        """
        self.hn = 1
        super().__init__(xc, R, posstates=None, enforceConstraint=False,
                         FOVradius=1, FOVsectorhalfangle=1, dirn=1, recordSensorState=False,**kwargs)

        self.sensStateStrs = ['r']


    def penaltyFOV(self,z, angFOVdev, rFOVdev):
        L = 0
        isinFOV = 1
        if rFOVdev > 0 or np.abs(angFOVdev) > self.FOVsectorhalfangle:
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

        r = np.sqrt((xkp[0] - self.xc[0])**2 + (xkp[1] - self.xc[1])**2)
        th = np.atan2(xkp[1] - self.xc[1], xkp[0] - self.xc[0])

        angFOVdev = self.dirn - th
        rFOVdev = r - self.FOVradius

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
                 FOVradius=1, FOVsectorhalfangle=1, dirn=1,recordSensorState=False, **kwargs):
        """
        enforceConstraint

        """
        self.hn = 1
        super().__init__(xc, R, posstates=None, enforceConstraint=False,
                         FOVradius=1, FOVsectorhalfangle=1, dirn=1,recordSensorState=False, **kwargs)

        self.sensStateStrs = ['th']


    def penaltyFOV(self,z, angFOVdev, rFOVdev):
        L = 0
        isinFOV = 1
        if rFOVdev > 0 or np.abs(angFOVdev) > self.FOVsectorhalfangle:
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

        r = np.sqrt((xkp[0] - self.xc[0])**2 + (xkp[1] - self.xc[1])**2)
        th = np.atan2(xkp[1] - self.xc[1], xkp[0] - self.xc[0])

        angFOVdev = self.dirn - th
        rFOVdev = r - self.FOVradius

        z = th
        isinFOV = 1
        L = 0
        if self.enforceConstraint:
            isinFOV, L = self.penaltyFOV(z, angFOVdev, rFOVdev)

        return [z, isinFOV, L]









