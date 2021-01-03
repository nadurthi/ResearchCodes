import numpy as np
from physmodels import motionmodels as phymm
from scipy.integrate import solve_ivp

class Duffing(phymm.MotionModel):
    motionModelName = 'Duffing'
    def __init__(self, a=3, b=1, **kwargs):
        self.a = a
        self.b = b
        self.fn = 2
        super().__init__(**kwargs)
    def contModel(self,t,x,a,b):
        dx=np.zeros(2)
        dx[0] = x[1]
        dx[1] = -a*x[0]-b*x[0]**3
        
        return dx
    def integrate(self,t_eval,x0):
        sol = solve_ivp(self.contModel, [t_eval[0], t_eval[-1]], x0,t_eval = t_eval,method='RK45',args=(self.a,self.b), rtol=1e-8, atol=1e-8)        
        return sol.y.T
    
    def integrate_batch(self,t_eval,X0):
        S=[]
        for x0 in X0:
            sol=solve_ivp(self.contModel, [t_eval[0], t_eval[-1]], x0,t_eval = t_eval,method='RK45',args=(self.a,self.b), rtol=1e-8, atol=1e-8)        
            S.append(sol.y.T)
        return np.stack(S,axis=0)
    
    
    def propforward(self, t, dt, xk, uk=0, **params):
        sol = solve_ivp(self.contModel, [t, t+dt], xk,method='RK45',args=(self.a,self.b), rtol=1e-8, atol=1e-8)        
        return (t + dt, sol.y.T[-1])

    def processNoise(self,t,dt,xk,uk=0,**params):
        
        Q = dt**2*np.array([[1,0],[0,1]])

        super().processNoise(**params)
        return Q

    def F(self, t, dt, xk,uk=0, **params):
        super().F(t, dt, xk, uk=None, **params)
        F=None

        return F
