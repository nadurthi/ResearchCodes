import scipy
import math
import numpy as np
from numpy import linalg as nplinalg
from scipy import linalg as sclinalg
import matplotlib.pyplot as plt
import pandas as pd
import ssatoolkit as ssatl
import ssatoolkit.constants as cst 
#import laplace_method as iod
from ssatoolkit import mht
from ssatoolkit import coords
from ssatoolkit import filters
from ssatoolkit.iod import laplace_method as lpmthd
from ssatoolkit.iod import doubler_iteration as dbriter
from ssatoolkit import ukf 
import ssatoolkit.propagators as ssaprop
from ssatoolkit.sensors import Sensor,SensorSet
from ssatoolkit.targets import GenOrbits,TrueTargets,CatalogTargets
import os
import plotting
import networkx as nx
import pickle as pkl

from ssatoolkit.od import differential_correction
# all simulations are with respect to fixed Earth
# so coordinate system is fixed


plt.close('all')



t0 = 0
tf = 24 * 3600  # for 1 day
dt = 1*60  # seconds
Tvec =  np.arange(t0, tf+dt, dt)



#%% sensors

sensors = SensorSet()
sensors.append(Sensor(idx=1, Rk=np.diag([0.0008**2,0.0008**2]), 
                      earthpos=[-23.02331*coords.to_rad(), -67.75379*coords.to_rad(), cst.Earth.radius], 
                      half_angle=60*coords.to_rad(),rmax=7000,orientation='normal'))
sensors.append(Sensor(idx=2, Rk=np.diag([0.0008**2,0.0008**2]), 
                      earthpos=[49.133949*coords.to_rad(), 1.441930*coords.to_rad(), cst.Earth.radius], 
                      half_angle=60*coords.to_rad(),rmax=7000,orientation='normal'))
sensors.append(Sensor(idx=3, Rk=np.diag([0.0008**2,0.0008**2]), 
                      earthpos=[35.281287*coords.to_rad(), -116.783051*coords.to_rad(), cst.Earth.radius], 
                      half_angle=60*coords.to_rad(),rmax=7000,orientation='normal'))







def plotsim(k0,k,ax,sensors,Data,estTargets,planet):
    # plot earth
    plotting.plotPlanet(ax,planet)
    
    # plot traj
    for key in Data.true_trajectory.keys():
        #print(true_trajectories[key][:, 0])
        ax.plot3D(Data.true_trajectory[key][k0:k, 0], 
                  Data.true_trajectory[key][k0:k, 1], Data.true_trajectory[key][k0:k, 2], 'black')


    #plot sensors
    for sens in sensors.itersensors():
        plotting.plot_cone(ax,sens.r_site,sens.orientation,
                  sens.rmax,sens.half_angle,sens.plotparams)
            
    
    # plot estimated target tracks
    if estTargets is not None:
        pass
    

genOrb = GenOrbits(cst.Earth.radius,cst.Earth.mu)
df = genOrb.genOrbits(100,sensors,Tvec,30,[7500,12000],[0.0001,0.9],[0.1,0.9*np.pi],[0.1,0.9*np.pi],[0.1,0.9*np.pi],[0,2*np.pi])
with open('data/randomOrbits.pkl','wb') as F:
    pkl.dump({'df':df,'Tvec':Tvec,'t0':t0,'dt':dt,'tf':tf,'sensors':sensors},F)
# genOrb.save('data/randomOrbits.csv')
#%% targets

Data = TrueTargets(cst.Earth.mu)
t0,dt,tf,Tvec = Data.get_pickle('data/randomOrbits.pkl')
# Data.getCSVcustom('data/randomOrbits.csv')

Data.generate_trajectory(Tvec)

# Data.getMeasStats(sensors,t0, tf, dt,NminMeas = 10)


figsim = plt.figure()
axsim = figsim.add_subplot(111, projection='3d')

estTargets=None
plotsim(0,2000,axsim,sensors,Data,estTargets,cst.Earth)


#%% test laplace method of iod
target_Ids = Data.gettargetIdxs()
targid =  target_Ids[50]
sens = sensors[2]
Tk,Zk = Data.get_meas_sens_target(targid,sens,list(range(len(Tvec))),Tvec,withNoise=False)




k1=120

ft=5
t1,t2,t3 = Tk[k1:k1+3*ft:ft]

T=Tk[k1:k1+3*ft:ft]
L=Zk[k1:k1+3*ft:ft]
Tchecks=Tk[k1-5:k1]
Lchecks=Zk[k1-5:k1]


def addNoise(zk,Rk):
    r,theta,phi = coords.cart2spherical(zk[0],zk[1],zk[2])
    anglenoise = np.matmul(sclinalg.sqrtm(Rk) , np.random.randn(2))
    # print(anglenoise)
    theta=theta+anglenoise[0]
    phi=phi+anglenoise[1]
    zk = coords.spherical2cart(r,theta,phi)
    return zk

withNoise=True
if withNoise:
    Rk=np.diag([(0.001*np.pi/180)**2,(0.001*np.pi/180)**2])
    for i in range(len(L)):
        L[i] = addNoise(L[i],Rk)
    for i in range(len(Lchecks)):
        Lchecks[i] = addNoise(Lchecks[i],Rk)
        
    
true_rv = Data.true_trajectory[targid][k1+ft, :]
true_rv2orb=[]
for k in range(k1-5,k1+3):
    trv = Data.true_trajectory[targid][k, :]
    orb= coords.from_RV_to_OE_vec(trv,cst.Earth.mu)
    true_rv2orb.append(orb)
true_rv2orb=np.array(true_rv2orb)

true_orb = Data.true_catalogue.loc[targid,['a','e','i','Om','om','f']].values

rvlap = lpmthd.laplace_orbit_fit(sens.r_site, T, L,Tchecks,Lchecks, cst.Earth.mu,20*coords.to_rad(),0.25)
# orblap= coords.from_RV_to_OE_vec(rvlap[0],cst.Earth.mu)

# if len(rvlap)>0:
#     orb = coords.from_RV_to_OE(rvlap[0][0:3],rvlap[0][3:6],cst.Earth.mu)
#     print(np.vstack([orb,true_orb]).T)


rvdbr = dbriter.double_r_iteration(sens.r_site, T, L,Tchecks,Lchecks, cst.Earth.mu,1e-1)

print("---------")
print("laplace = ",true_rv-rvlap)
if rvdbr is not None:
    print("dbr = ",true_rv-rvdbr)
    orb = coords.from_RV_to_OE(rvdbr[0:3],rvdbr[3:6],cst.Earth.mu)

#%%
    # print(true_orb)
# if rv.shape[0]>0:
#     orbelems=[]
#     for i in range(rv.shape[0]):
#         orb = coords.from_RV_to_OE(rv[i][0:3],rv[i][3:6],cst.Earth.mu)
#         orbelems.append(orb)

dct = coords.DimensionalConverter(cst.Earth)
# rvdbr_can = dct.true2can_posvel(rvdbr)  
# t1_can = dct.true2can_time(T[1])

data=np.zeros((len(Zk),7))
# data_can=np.zeros_like(data)
for i in range(len(Zk)):
    data[i,0:3] = Zk[i]
    data[i,3:6] = sens.r_site
    data[i,6] = Tk[i]
    
    # data_can[i,0:3] = Zk[i]
    # data_can[i,3:6] = dct.true2can_pos(sens.r_site)
    # data_can[i,6] = dct.true2can_time(Tk[i])

orblap= coords.from_RV_to_OE(rvlap[0][:3],rvlap[0][3:],cst.Earth.mu)
orb0 = [8000,orblap[1],orblap[2],orblap[3],orblap[4],orblap[5]]
res = differential_correction.orbit_det(T[1],rvdbr,data[::2],cst.Earth.mu)
# res_can = differential_correction.orbit_det(t1_can,rvdbr_can,data_can,1)
errs=differential_correction.zkDistErrs(res['x'],T[1], data,cst.Earth.mu)
print(np.mean(errs))
#%%
orbtrue=Data.true_catalogue.loc[targid,['a','e','i','Om','om','f']].values
rvtrue = coords.from_OE_to_RV(orbtrue,cst.Earth.mu)
ttrue = Tvec[0]
tf=Tvec[-1]
trajtrue=ssaprop.propagate_orb(Tvec, orbtrue,cst.Earth.mu)

Data.true_trajectory[targid]-trajtrue

_,X1=ssaprop.propagate_rv_tvec(Tvec[2000], trajtrue[2000], Tvec, cst.Earth.mu)
_,X2=ssaprop.propagate_FnG_mixed(Tvec[2000], trajtrue[2000], Tvec, cst.Earth.mu)
print(np.max(np.abs(X1-trajtrue)),np.max(np.abs(X2-trajtrue)))


#%% Check if orbital element conversions are correnct
k=200
targid =0

rv = Data.true_trajectory[targid][k, :]
true_rv = Data.true_trajectory[targid][k, :]
true_orb = Data.true_catalogue.loc[targid,['a','e','i','Om','om','f']].values

true_orb2rv = coords.from_OE_to_RV(true_orb,cst.Earth.mu)

Orbit1 = coords.from_RV_to_OE_vec(rv,cst.Earth.mu)
rvback = coords.from_OE_to_RV(Orbit1,cst.Earth.mu)

np.vstack([rv,rvback])
