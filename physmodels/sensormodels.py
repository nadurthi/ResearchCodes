# -*- coding: utf-8 -*-


class SensorModel:
    def __init__(self, **kwargs):
        pass

    def updateparam(self, **params):
        pass

    def measNoise(self, t, dt, xk, **params):
        return self.R

    def H(self, t, dt, xk, **params):
        pass


class DiscLTSensorModel:
    def __init__(self, H, R, **kwargs):
        self.H = H
        self.R = R

    def measNoise(self, t, dt, xk, **params):
        return self.R

    def H(self, t, dt, xk, **params):
        return self.H
    
    def __call__(t, dt, xk):
        zk = np.matmul(self.H,xk)
        return zk
    

class PlanarSensorModel(SensorModel):
    def __init__(
            self,
            xc,
            R,
            posstates=None,
            enforceConstraint=False,
            rmax=1,
            alpha=1,
            phi=1,
            **kwargs):
        self.rmax = rmax  # FOV
        self.alpha = alpha  # FOV
        self.xc = xc
        self.R = R  # noise covariance
        self.phi = phi  # look direction
        self.posstates = posstates

        self.enforceConstraint = enforceConstraint
        super(PlanarSensorModel, self).__init__(**kwargs)

    def updateparam(self, **params):
        self.rmax = params.get('rmax', self.rmax)
        self.alpha = params.get('alpha', self.alpha)
        self.xc = params.get('xc', self.xc)
        self.R = params.get('R', self.R)
        self.phi = params.get('phi', self.phi)

        super(PlanarSensorModel, self).updateparam(**kwargs)


class RcircularFOVsensor(PlanarSensorModel):
    """
    [r]
    """

    def __init__(
            self,
            xc,
            R,
            posstates=None,
            enforceConstraint=False,
            rmax=1,
            alpha=1,
            phi=1,
            **kwargs):
        """
        enforceConstraint

        """
        self.hn = 1
        super(
            RcircularFOVsensor,
            self).__init__(
            xc,
            R,
            posstates=None,
            enforceConstraint=False,
            rmax=1,
            alpha=1,
            phi=1,
            **kwargs)

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

    def __init__(
            self,
            xc,
            R,
            posstates=None,
            enforceConstraint=False,
            rmax=1,
            alpha=1,
            phi=1,
            **kwargs):
        """
        enforceConstraint

        """
        self.hn = 1
        super(
            RcircularFOVsensor,
            self).__init__(
            xc,
            R,
            posstates=None,
            enforceConstraint=False,
            rmax=1,
            alpha=1,
            phi=1,
            **kwargs)

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

    def __init__(
            self,
            xc,
            R,
            posstates=None,
            enforceConstraint=False,
            rmax=1,
            alpha=1,
            phi=1,
            **kwargs):
        """
        enforceConstraint

        """
        self.hn = 2
        super(
            RcircularFOVsensor,
            self).__init__(
            xc,
            R,
            posstates=None,
            enforceConstraint=False,
            rmax=1,
            alpha=1,
            phi=1,
            **kwargs)

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
