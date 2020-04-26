import logging
import numpy as np
import uuid

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MotionModel:
    def __init__(self, *args, **kwargs):
        self.ID = uuid.uuid4()

    def propforward(self, *args, uk=None, **kwargs):
        pass

    def processNoise(self, *args, **kwargs):
        pass

    def F(self, t, dt, xk, uk=None, **params):
        pass

# autopep8 --in-place --aggressive --aggressive -v -r .


class DiscLTIstateSpaceModel(MotionModel):
    def __init__(self, A, B=None, C=None, D=None, Q=None, **kwargs):
        self.A = A
        self.fn = A.shape[0]

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

        return xk1

    def processNoise(self, t, dt, xk, uk=0, **params):
        return self.Q


class KinematicModel_CT(MotionModel):
	def __init__(self, L1=0.16, L2=0.001, **kwargs):
		self.L1 = L1
		self.L2 = L2
        self.fn = 5

        super(KinematicModel_CT, self).__init__(**kwargs)

	def propforward(self, t, dt, xk, uk=0, **params):
		T = dt;
        omg = xk[-1] + 1e-10

        xk1 = np.matmul(np.array([[1, 0, np.sin(omg * T) / omg, -(1 - np.cos(omg * T)) / omg, 0],
             [0, 1, (1 - np.cos(omg * T)) / omg, np.sin(omg * T) / omg, 0],
             [0, 0, np.cos(omg * T), -np.sin(omg * T), 0],
             [0, 0, np.sin(omg * T), np.cos(omg * T), 0],
             [0, 0, 0, 0, 1]]), xk);
        super(KinematicModel_CT, self).propforward(**params)

        return (t + dt, xk1)


    def processNoise(self,t,dt,xk,uk=0,**params):
        T = dt
        Q_CT = self.L1 * np.array([[T**3 / 3, 0, T**2 / 2, 0, 0],
			[0, T**3 / 3, 0, T**2 / 2, 0], [
        	T**2 / 2, 0, T, 0, 0], [
		    0, T**2 / 2, 0, T, 0], [
        	0, 0, 0, 0, T * self.L2 / self.L1]])
        
        super(KinematicModel_CT,self).processNoise(**params)
        return Q_CT
    
    def F(self, t, dt, xk,uk=0, **params):
        super().F(t, dt, xk, uk=None, **params)

    
class KinematicModel_UM(MotionModel):
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
        
        super(KinematicModel_UM,self).processNoise(**params)
        return Q_UM
   
   def F(self,t,dt,xk,uk=0,**params):
       
       T = dt
       F = np.array([[1, 0, T, 0],
                      [0, 1, 0, T],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
       super().F(t, dt, xk, uk=None, **params)   
       return F
