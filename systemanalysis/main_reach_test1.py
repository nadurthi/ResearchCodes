# -*- coding: utf-8 -*-

import time
import numpy as np
import numpy.linalg as nplalg
import scipy.linalg as sclalg
import matplotlib.pyplot as plt
from physmodels import duffing as phymdf
import matplotlib.colors as mcolors
# matplotlib.colors
# matplotlib.colors.PowerNorm
# matplotlib.axes.Axes.hist2d
# matplotlib.pyplot.hist2d
plt.close('all')
from scipy.integrate import solve_ivp
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from uq.information import distance as uqinfodist
from uq.stats import moments as uqstatmom
import uq.stats.pdfs as uqstpdf
import uq.quadratures.cubatures as uqcub
import uq.gmm.gmmbase as uqgmm
#%%
# mdl = phymdf.DiscDuffingControlStack()
mdl = phymdf.VanderpolOscillator(mu=0.8)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
X = mdl.integrate(np.linspace(0,10,100),np.array([2,2]))
ax1.plot(X[:,0],X[:,1])
plt.show()


lb  = np.array([-3.5,-3.5])
ub  = np.array([3.5,3.5])
dt=0.1

# N=20000
XV,YV=np.meshgrid(np.linspace(lb[0],ub[0],100),np.linspace(lb[1],ub[1],100))
X0 = np.hstack([XV.reshape(-1,1),YV.reshape(-1,1)])
N = X0.shape[0]
# X0 = np.random.rand(N,2)
# X0 = X0*(ub-lb)+lb


fig = plt.figure()
ax = fig.add_subplot(111)

Nt = 100
Tvec = np.linspace(0,10,Nt)
dt = Tvec[1]-Tvec[0]

ax1.plot(X0[:, 0], X0[:, 1],'r.')
cnorm = mcolors.PowerNorm(0.5)

H=[]
cb=None
Xk1=X0.copy()
for i in range(Nt):
    ax.cla()
    if cb is not None:
        cb.remove()
        
    h=ax.hist2d(Xk1[:, 0], Xk1[:, 1], bins=25, range =[[lb[0],ub[0]],[lb[1],ub[1]]], 
                                               norm=mcolors.PowerNorm(0.5),density=True)
    # ax.plot(Xk1[:, 0], Xk1[:, 1],'b.')
    ax.set_title(" i = %d"%i)
    cb = fig.colorbar(h[3], ax=ax)
    fig.savefig('VanderpolDensity_%03d.eps'%i, format='eps', dpi=1200)
    
    plt.show()
    plt.pause(0.2)
    H.append(h[0].reshape(1,-1))
    
    tt,Xk1 = mdl.propforward_batch( Tvec[i], dt, Xk1, uk=0)
    Xk1 = Xk1+0.01*np.random.randn(N,2)
    
    
    
H = np.vstack(H)

#%%
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Tvec,H[:,110],'b',linewidth=2)
ax.set_xlabel('t (s)',fontsize=20)
ax.set_ylabel('p($B_i$)',fontsize=20)
ax.grid()
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=12)

#%%
#  136, 
# fig = plt.figure()
ax.cla()
ax = fig.add_subplot(111)
ax.plot(Tvec,H[:,133],'b',linewidth=2)
ax.set_xlabel('t (s)',fontsize=20)
ax.set_ylabel('p($B_i$)',fontsize=20)
ax.grid()
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=12)


#%%
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(0,5000):
    ax.cla()
    ax.plot(Tvec,H[:,i],'b',linewidth=2)
    ax.set_xlabel('t (s)',fontsize=20)
    ax.set_ylabel('p($B_i$)',fontsize=20)
    ax.set_title(" i = %d"%i)
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    
    plt.show()
    plt.pause(0.5)



#%% Information metrics for linear systems and duffing oscillator.
plt.close("all")
# linear system
Q=0.01*np.identity(2)
A = 0.5*np.array([[-1,2],[-3,-1]])
P = 5**2*np.identity(2)
m = np.zeros(2)
t0=0
tf=20


def contModel(t,x):
    dx=A.dot(x)
    return dx

def contModelcov(t,P):
    P=P.reshape(2,2)
    dP = np.multiply(A,P)+np.multiply(P,A.T)+Q
    return dP.reshape(-1) 


solm = solve_ivp(contModel, [t0, tf], m,t_eval = np.linspace(t0,tf,500),method='RK45',args=(), rtol=1e-12, atol=1e-12)        
mt = solm.y.T

solP = solve_ivp(contModelcov, [t0, tf], P.reshape(-1),t_eval = np.linspace(t0,tf,500),method='RK45',args=(), rtol=1e-12, atol=1e-12)        
Pt = solP.y.T

def entropy(m,P):
    return 0.5*np.log(nplalg.det(2*np.pi*np.e*P))

def KL(m0,P0,m1,P1):
    d=m1-m0
    k=len(m0)
    return 0.5*( np.trace( np.matmul(nplalg.inv(P1),P0) )+ d.dot( nplalg.inv(P1).dot(d) )-k+np.log(nplalg.det(P1)/nplalg.det(P0)) )

def WassersteinDist(m1,P1,m2,P2):
    s1 = sclalg.sqrtm(P1)
    g = np.matmul(np.matmul(s1,P2),s1) 
    d = np.sqrt( nplalg.norm(m1-m2)+np.trace(  P1+P2-2*sclalg.sqrtm( g  ) ) )
    
    return d
    
E = np.zeros(len(solm.t))
K = np.zeros(len(solm.t))
W = np.zeros(len(solm.t))
P0 = 0.001**2*np.identity(2)
m0 = np.zeros(2)

for i in range(len(solm.t)):
    pp = Pt[i].reshape(2,2)
    mm = mt[i]
    E[i] =  entropy(mm,pp)
    # K[i] =  KL(mm,pp,m0,P0)
    K[i] =  KL(m0,P0,mm,pp)
    W[i] =  WassersteinDist(m0,P0,mm,pp)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(mt[:,0],mt[:,1],label='Entropy')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(solm.t,E,label='Entropy')
ax.legend()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(solm.t,K,label='KL')
ax.legend()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(solm.t,W,label='Wasserstein')
ax.legend()

plt.show()

# 3D plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X0 = np.arange(-3, 3, 0.1)
Y0 = np.arange(-3, 3, 0.1)
X0, Y0 = np.meshgrid(X0, Y0)
        
Z0 = np.zeros_like(X0)
for i in range(X0.shape[0]):
    for j in range(X0.shape[1]):
        Z0[i,j] = uqstpdf.gaussianPDF(np.array([X0[i,j],Y0[i,j]]),m0,P0)


        
        
# Plot the surface.
surf = ax.plot_surface(X0, Y0, Z0, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)



# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# # A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# 3D plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.


tt=175
pp = Pt[tt].reshape(2,2)
mm = mt[tt]
        

X = np.arange(-5, 5, 0.05)
Y = np.arange(-5, 5, 0.05)
X, Y = np.meshgrid(X, Y)

Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i,j] = uqstpdf.gaussianPDF(np.array([X[i,j],Y[i,j]]),mm,pp)
        
        
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, FaceColor='b',
                       linewidth=0, antialiased=False)



plt.show()

#%% Gaussian mixture for a stable for duffingoscillator
plt.close("all")
# linear system
Q=0.01*np.identity(2)

Pint = 0.1**2*np.identity(2)

X = np.linspace(-3, 3, 10)
Y = np.linspace(-3, 3, 10)
X, Y = np.meshgrid(X, Y)
m=[]
P=[]
Xquad=[]
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        mm=[X[i,j],Y[i,j]]
        m.append(mm)
        P.append(Pint)
        XX,w = uqcub.UT_sigmapoints(mm, Pint)
        Xquad.append( XX )


t0=0
tf=50
Nt=100
Ncomp = len(Xquad)

def contModel(t,x):
    dx=np.zeros(2)
    dx[0] = x[1]
    dx[1] = 1*x[0]-1*x[0]**3-1*x[1]
    return dx

solm = solve_ivp(contModel, [t0, tf], [-3,3],t_eval = np.linspace(t0,tf,Nt),method='RK45',args=(), rtol=1e-12, atol=1e-12)        
ss=solm.y.T 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ss[:,0],ss[:,1])
plt.show()    
        
for ng in range(len(Xquad)):
    S=[]
    for i in range(len(Xquad[ng])):
        solm = solve_ivp(contModel, [t0, tf], Xquad[ng][i],t_eval = np.linspace(t0,tf,Nt),method='RK45',args=(), rtol=1e-12, atol=1e-12)        
        S.append( solm.y.T )
    S=np.stack(S,axis=0)
    Xquad[ng] = S


gmm0 = uqgmm.GMM.fromlist([np.array([1,0]),np.array([-1,0])],[0.0001**2*np.identity(2),0.0001**2*np.identity(2)],np.ones(2)/2,0)
W=np.zeros(len(solm.t))
for ti in range(len(solm.t)):
    print(ti)
    MM=[]
    PP=[]
    for ng in range(len(Xquad)):
        x=Xquad[ng][:,ti,:]
        mm,pp = uqstatmom.MeanCov(x,w)
        MM.append(mm)
        PP.append(pp)
    gmmti = uqgmm.GMM.fromlist(MM,PP,np.ones(Ncomp)/Ncomp,solm.t[ti])
    W[ti]=uqinfodist.wassersteinrDistGMM(gmm0,gmmti) 
    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(solm.t,W,label='Wasserstein')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
for ng in range(len(Xquad)):
    ax.plot(Xquad[ng][0,0,0],Xquad[ng][0,0,1],'bo')
    ax.plot(Xquad[ng][0,-1,0],Xquad[ng][0,-1,1],'r.')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
for ng in range(len(Xquad)):
    ax.plot(Xquad[ng][0,:,0],Xquad[ng][0,:,1])
plt.show()

#%% Gaussian mixture for a stable for duffingoscillator
plt.close("all")
# linear system
Q=0.01*np.identity(2)

Pint = 0.25**2*np.identity(2)

X = np.linspace(-3, 3, 10)
Y = np.linspace(-3, 3, 10)
X, Y = np.meshgrid(X, Y)
m=[]
P=[]
Xquad=[]
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        mm=[X[i,j],Y[i,j]]
        m.append(mm)
        P.append(Pint)
        XX,w = uqcub.UT_sigmapoints(mm, Pint)
        Xquad.append( XX )


t0=0
tf=50
Nt=500
Ncomp = len(Xquad)

def contModel(t,x):
    dx=np.zeros(2)
    dx[0] = x[1]
    dx[1] = 3*(1-x[0]**2)*x[1]-x[0]
    
    return dx


solm = solve_ivp(contModel, [t0, tf], [1,1],t_eval = np.linspace(t0,tf,Nt),method='RK45',args=(), rtol=1e-5, atol=1e-5)        
ss=solm.y.T 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ss[:,0],ss[:,1])
plt.show()    
        
for ng in range(len(Xquad)):
    print(ng)
    S=[]
    for i in range(len(Xquad[ng])):
        solm = solve_ivp(contModel, [t0, tf], Xquad[ng][i],t_eval = np.linspace(t0,tf,Nt),method='RK45',args=(), rtol=1e-12, atol=1e-12)        
        S.append( solm.y.T )
    S=np.stack(S,axis=0)
    Xquad[ng] = S


gmm0 = uqgmm.GMM.fromlist([np.array([0,0])],[0.0001**2*np.identity(2)],np.ones(1)/1,0)
W=np.zeros(len(solm.t))
E=np.zeros(len(solm.t))
for ti in range(len(solm.t)):
    print(ti)
    MM=[]
    PP=[]
    for ng in range(len(Xquad)):
        x=Xquad[ng][:,ti,:]
        mm,pp = uqstatmom.MeanCov(x,w)
        MM.append(mm)
        PP.append(pp)
    gmmti = uqgmm.GMM.fromlist(MM,PP,np.ones(Ncomp)/Ncomp,solm.t[ti])
    W[ti]=uqinfodist.wassersteinrDistGMM(gmm0,gmmti) 
    xp=gmmti.random(100000)
    E[ti]= -np.mean(np.log(gmmti.pdf(xp)))
    

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(solm.t,W,label='Wasserstein')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(solm.t,W,label='Entropy')
ax.set_legend()
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
for ng in range(len(Xquad)):
    ax.plot(Xquad[ng][0,0,0],Xquad[ng][0,0,1],'bo')
    ax.plot(Xquad[ng][0,-1,0],Xquad[ng][0,-1,1],'r.')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
for ng in range(len(Xquad)):
    ax.plot(Xquad[ng][0,:,0],Xquad[ng][0,:,1])
plt.show()


# entropy



