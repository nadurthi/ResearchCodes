import logging
import numpy as np
import uuid
from scipy.linalg import block_diag
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MotionModel:
	motionModelName = 'MotionModel'
	def __init__(self, *args, **kwargs):
		self.ID = uuid.uuid4()

	def propforward(self, uk=None, *args, **kwargs):
		pass

	def processNoise(self, *args, **kwargs):
		pass

	def F(self, t, dt, xk, uk=None, **params):
		pass

	def debugStatus(self):
		ss=[]
		ss.append(['----------','----------'])
		ss.append(['MotionModel:',self.ID])
		ss.append(['----------','----------'])
		ss.append(['ModelName:',self.motionModelName])
		ss.append(['fn',self.fn])

		return ss
# autopep8 --in-place --aggressive --aggressive -v -r .

class MultipleMarkovMotionModel:
    def __init__(self,models,p):
        """
        p is the transition matrix: square matrix from i to j
        p[i,j] is from i to j
        """
        self.ID = uuid.uuid4()
#        self.Nmodels = len(models)
        self.models = models
        self.p = p

    @property
    def Nmodels(self):
        return len(self.models)

    def __getitem__(self,i):
        return self.models[i]



class DiscLTIstateSpaceModel(MotionModel):
    motionModelName = 'DiscLTIstateSpaceModel'
    def __init__(self, A, B=None, C=None, D=None, Q=None, **kwargs):
        self.fn = A.shape[0]
        self.A = A
        if B is None:
            self.B = np.zeros((self.fn, 1))
        if C is None:
            self.C = np.eye(self.fn)

        if D is None:
            self.D = np.zeros((self.fn, 1))
        if Q is None:
            self.Q = np.zeros((self.fn, self.fn))
            self.sQ = np.zeros((self.fn, self.fn))

        super().__init__(**kwargs)

    def F(self, t, dt, xk, uk=0, **params):
        super().F(t, dt, xk, uk=None, **params)
        return self.A

    def propforward(self, t, dt, xk, uk=0, **params):
        xk1 = np.matmul(self.A, xk) + np.matmul(self.B, uk)
        return (t + dt, xk1)
    def processNoise(self, t, dt, xk, uk=0, **params):
        return self.Q


class KinematicModel_CT(MotionModel):
    motionModelName = 'KinematicModel_CT'
    def __init__(self, L1=0.16, L2=0.001, **kwargs):
        self.L1 = L1
        self.L2 = L2
        self.fn = 5
        super().__init__(**kwargs)
    def propforward(self, t, dt, xk, uk=0, **params):
        T = dt;
        omg = xk[-1] + 1e-10

        xk1 = np.matmul(np.array([[1, 0, np.sin(omg * T) / omg, -(1 - np.cos(omg * T)) / omg, 0],
				 [0, 1, (1 - np.cos(omg * T)) / omg, np.sin(omg * T) / omg, 0],
				 [0, 0, np.cos(omg * T), -np.sin(omg * T), 0],
				 [0, 0, np.sin(omg * T), np.cos(omg * T), 0],
				 [0, 0, 0, 0, 1]]), xk);
        super().propforward(**params)
        return (t + dt, xk1)

    def processNoise(self,t,dt,xk,uk=0,**params):
        T = dt
        Q_CT = self.L1 * np.array([[T**3 / 3, 0, T**2 / 2, 0, 0],
		 [0, T**3 / 3, 0, T**2 / 2, 0], [
		 T**2 / 2, 0, T, 0, 0], [
			0, T**2 / 2, 0, T, 0], [
		 0, 0, 0, 0, T * self.L2 / self.L1]])

        super().processNoise(**params)
        return Q_CT

    def F(self, t, dt, xk,uk=0, **params):
        super().F(t, dt, xk, uk=None, **params)
        T = dt;
        omg = xk[-1] + 1e-10

        if np.abs(omg) >1e-2:

            f=[ np.cos(omg*T)*T*xk[2]/omg - np.sin(omg*T)*xk[2]/omg**2 - np.sin(omg*T)*T*xk[3]/omg - (-1+np.cos(omg*T))*xk[3]/omg**2,
            np.sin(omg*T)*T*xk[2]/omg - (1-np.cos(omg*T))*xk[2]/omg**2 + np.cos(omg*T)*T*xk[3]/omg - np.sin(omg*T)*xk[3]/omg**2,
            -np.sin(omg*T)*T*xk[2] - np.cos(omg*T)*T*xk[3],
            np.cos(omg*T)*T*xk[2] - np.sin(omg*T)*T*xk[3] ]

            F = np.array([[1, 0, np.sin(omg * T) / omg, -(1 - np.cos(omg * T)) / omg, f[0] ],
				 [0, 1, (1 - np.cos(omg * T)) / omg, np.sin(omg * T) / omg, f[1]],
				 [0, 0, np.cos(omg * T), -np.sin(omg * T), f[2]],
				 [0, 0, np.sin(omg * T), np.cos(omg * T), f[3]],
				 [0, 0, 0, 0, 1]])

        else:
            F= [ [1,0,T,0,-0.5*T**2*xk[3]],
                 [0,1,0,T,0.5*T**2*xk[2]],
                 [0,0,1,0,-T*xk[3]],
                 [0,0,0,1,T*xk[2]],
                 [0,0,0,0,1]
                 ]
            F = np.array(F)

        return F


class KinematicModel_UM(MotionModel):
	motionModelName = 'KinematicModel_UM'
	def __init__(self, L1=0.16,L2=0.001,**kwargs):
		self.L1 = L1
		self.L2 = L2
		self.fn = 4

		super(KinematicModel_UM,self).__init__(**kwargs)

	def propforward(self, t, dt, xk,uk=0, **params):
		T = dt;

		xk1 = np.matmul(np.array([[1, 0, T, 0],
									[0, 1, 0, T],
									[0, 0, 1, 0],
									[0, 0, 0, 1]]), xk);



		super(KinematicModel_UM,self).propforward(**params)
		return (t+dt,xk1)

	def processNoise(self,t,dt,xk,uk=0,**params):

		T = dt
		Q_UM=self.L1 * np.array([[T**3 / 3, 0, T**2 / 2, 0],
		 [0, T**3 / 3, 0, T**2 / 2], [
		 T**2 / 2, 0, T, 0], [
			0, T**2 / 2, 0, T]])

		super().processNoise(**params)
		return Q_UM

	def F(self,t,dt,xk,uk=0,**params):
		T = dt
		F = np.array([[1, 0, T, 0],
									[0, 1, 0, T],
									[0, 0, 1, 0],
									[0, 0, 0, 1]])
		super().F(t, dt, xk, uk=uk, **params)
		return F

class KinematicModel_UM_5state(KinematicModel_UM):
    motionModelName = 'KinematicModel_UM_5state'
    def __init__(self, L1=0.16,L2=0.001,**kwargs):
        super().__init__(L1=L1,L2=L2,**kwargs)
        self.fn = 5

    def propforward(self, t, dt, xk,uk=0, **params):
        _,xk1 = super().propforward(t, dt, xk[0:4],uk=uk, **params)
        xk1 = np.hstack([xk1,0])
        return (t+dt,xk1)

    def processNoise(self,t,dt,xk,uk=0,**params):
        Q_UM = super().processNoise(t,dt,xk,uk=uk,**params)

        return block_diag(Q_UM,0)

    def F(self,t,dt,xk,uk=0,**params):
        F = super().F(t,dt,xk,uk=uk,**params)
        return block_diag(F,0)
