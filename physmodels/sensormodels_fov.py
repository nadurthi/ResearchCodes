import numpy as np
from physmodels.sensormodels import SensorModel

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

        self.sensStateStrs = ['x','y']
        self.sensstates = ['lRg', 'ltg', 'rmax', 'alpha'] # states required to make the measurement as time time t

        super().__init__(recordSensorState=False,**kwargs)

    def updateparam(self, updatetk, **params):
        self.rmax = params.get('rmax', self.rmax)
        self.alpha = params.get('alpha', self.alpha)
        self.xc = params.get('xc', self.xc)
        self.R = params.get('R', self.R)
        self.phi = params.get('phi', self.phi)

        self.lRg = np.array([[np.cos(self.phi), -np.sin(self.phi)],
                             [np.sin(self.phi), np.cos(self.phi)]])
        self.ltg = self.xc

        super(PlanarSensorModel, self).updateparam(updatetk, **params)


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

        self.sensStateStrs = ['r']


    def penaltyFOV(self,z, angFOVdev, rFOVdev):
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

        r = np.sqrt((xkp[0] - self.xc[0])**2 + (xkp[1] - self.xc[1])**2)
        th = np.atan2(xkp[1] - self.xc[1], xkp[0] - self.xc[0])

        angFOVdev = self.phi - th
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

        self.sensStateStrs = ['th']


    def penaltyFOV(self,z, angFOVdev, rFOVdev):
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

        r = np.sqrt((xkp[0] - self.xc[0])**2 + (xkp[1] - self.xc[1])**2)
        th = np.atan2(xkp[1] - self.xc[1], xkp[0] - self.xc[0])

        angFOVdev = self.phi - th
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
        self.sensStateStrs = ['r','th']

    def penaltyFOV(self,z, angFOVdev, rFOVdev):
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

        r = np.sqrt((xkp[0] - self.xc[0])**2 + (xkp[1] - self.xc[1])**2)
        th = np.atan2(xkp[1] - self.xc[1], xkp[0] - self.xc[0])

        angFOVdev = self.phi - th
        rFOVdev = r - self.rmax

        z = np.array([r, th])
        isinFOV = 1
        L = 0
        if self.enforceConstraint:
            isinFOV, L = self.penaltyFOV(z, angFOVdev, rFOVdev)

        return [z, isinFOV, L]





