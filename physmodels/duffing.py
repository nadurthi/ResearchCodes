import numpy as np
from physmodels import motionmodels as phymm
from scipy.integrate import solve_ivp

class VanderpolOscillator(phymm.MotionModel):
    motionModelName = 'VanderpolOscillator'
    def __init__(self, mu=3, **kwargs):
        self.mu = mu
        
        self.fn = 2
        super().__init__(**kwargs)
    def contModel(self,t,x,mu):
        dx=np.zeros(2)
        dx[0] = x[1]
        dx[1] = mu*(1-x[0]**2)*x[1]-x[0]
        
        return dx
    def integrate(self,t_eval,x0):
        sol = solve_ivp(self.contModel, [t_eval[0], t_eval[-1]], x0,t_eval = t_eval,method='RK45',args=(self.mu,), rtol=1e-8, atol=1e-8)        
        return sol.y.T
    
    def integrate_batch(self,t_eval,X0):
        S=[]
        for x0 in X0:
            sol=solve_ivp(self.contModel, [t_eval[0], t_eval[-1]], x0,t_eval = t_eval,method='RK45',args=(self.mu,), rtol=1e-8, atol=1e-8)        
            S.append(sol.y.T)
        return np.stack(S,axis=0)
    
    
    def propforward(self, t, dt, xk, uk=0, **params):
        sol = solve_ivp(self.contModel, [t, t+dt], xk,method='RK45',args=(self.mu,), rtol=1e-8, atol=1e-8)        
        return (t + dt, sol.y.T[-1])
    
    def propforward_batch(self, t, dt, Xk, uk=0, **params):
        t_eval = np.array([t,t+dt])
        S = self.integrate_batch(t_eval,Xk)
        return (t + dt, S[:,-1,:])
    
    def processNoise(self,t,dt,xk,uk=0,**params):
        
        Q = dt**2*np.array([[1,0],[0,1]])

        super().processNoise(**params)
        return Q

    def F(self, t, dt, xk,uk=0, **params):
        super().F(t, dt, xk, uk=None, **params)
        F=None

        return F
    
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
    
    def propforward_cov(self, t, dt, Pk, Q,uk=0, **params):
        A
        sol = solve_ivp(self.contModel, [t, t+dt], xk,method='RK45',args=(self.a,self.b), rtol=1e-8, atol=1e-8)        
        return (t + dt, sol.y.T[-1])
    
    def propforward_batch(self, t, dt, Xk, uk=0, **params):
        t_eval = np.array([t,t+dt])
        S = self.integrate_batch(t_eval,Xk)
        return (t + dt, S[:,-1,:])
    
    def processNoise(self,t,dt,xk,uk=0,**params):
        
        Q = dt**2*np.array([[1,0],[0,1]])

        super().processNoise(**params)
        return Q

    def F(self, t, dt, xk,uk=0, **params):
        super().F(t, dt, xk, uk=None, **params)
        F=None

        return F

class DiscDuffingControlStack(phymm.MotionModel):
    motionModelName = 'DiscDuffingControlStack'
    def __init__(self, a=3, b=1, **kwargs):
        self.a = a
        self.b = b
        self.fn = 2
        super().__init__(**kwargs)
    def discModel(self,dt,x,a,b):
        xk1=np.zeros(4)
        xk1[0] = x[0]+dt*x[1] +x[2]
        xk1[1] = x[1]-a*x[0]*dt-b*x[0]**3*dt+x[3]
        xk1[2] = x[2]
        xk1[3] = x[3]
        
        return xk1
    

    
    
    def propforward(self, t, dt, xk, uk=0, **params):
        xk1 = self.discModel(self,dt,xk,self.a,self.b)
        return (t + dt, xk1)
    
    def propforward_batch(self, t, dt, Xk, uk=0, **params):
        Xk1 = np.zeros_like(Xk)
        for i in range(Xk.shape[0]):
            Xk1[i] = self.discModel(self,dt,Xk[i],self.a,self.b)
        return (t + dt, Xk1)
    

    def F(self, t, dt, xk,uk=0, **params):
        super().F(t, dt, xk, uk=None, **params)
        F= np.array([[1,dt,1,0],[-self.a*dt-3*dt*self.b*x[0]**2,1,0,1],[0,0,1,0],[0,0,0,1]])

        return F