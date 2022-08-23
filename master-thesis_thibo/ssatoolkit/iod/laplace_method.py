import math
import numpy as np
# import toolkit as tl
# import radar_filter
from ssatoolkit import propagators
# import pandas as pd
# import os

from numpy import linalg as nplinalg

#omega_e = np.array([0, 0, 7.292115 * 10 ** (-5)]) # omega_earth   REPLACE IF EARTH ROTATION
# mu = 398600  # Km3/s2
def laplace_orbit_fit(r_site, T, L, Tchecks,Lchecks,mu,angleThres,ThresVotefrac):
    """ Determine the orbit using the laplace method.

    Parameters
    ----------

    r_site: numpy.array (3x1)
      Observation site's coordinates [xyz].
    T: numpy.array (3x1)
      Gives the time since the first observation.
    L: numpy.array (3x3)
      3 unit vectors that give a direction of observation.(each row is one observation)
      
    Tchecks,Lchecks: this is used to check at multiple observations
        ThresVotefrac:  vote to use to select the right solution
    
    angleThres: threshold for closeness of solution to  observation
    
    
    
    Returns
    -------

    RV: numpy.array
      One array that comprise position and velocity at one point.

    """

    # We build the RVA vector that comprise the position, speed, and acceleration of the observation site
    RVA = np.array([r_site])
    RVA = np.append(RVA, [0, 0, 0])#RVA = np.append(RVA, np.cross(omega_e, RVA[0:3]))   REPLACE IF EARTH ROTATION
    RVA = np.append(RVA, [0, 0, 0])#RVA = np.append(RVA, np.cross(omega_e, RVA[3:6]))   REPLACE IF EARTH ROTATION

    # Ldot and Ldot2 at the mid point:
    Ldot = (T[1]-T[2])/((T[0]-T[1])*(T[0]-T[2]))*L[0] + (2*T[1]-T[0]-T[2])/((T[1]-T[0])*(T[1]-T[2]))*L[1] + (T[1]-T[0])/((T[2]-T[0])*(T[2]-T[1]))*L[2]
    Ldot2 = 2*L[0]/((T[0]-T[1])*(T[0]-T[2])) + 2*L[1]/((T[1]-T[0])*(T[1]-T[2])) + 2*L[2]/((T[2]-T[0])*(T[2]-T[1]))

    # Computing all the determinants and needed stuff
    R = np.linalg.norm(RVA[:3])
    C = np.dot(L[1], RVA[:3])
    D = 2*np.linalg.det(np.vstack((L[1], Ldot, Ldot2)))
    
    D1 = np.linalg.det(np.vstack((L[1], Ldot, RVA[-3:])))
    D2 = np.linalg.det(np.vstack((L[1], Ldot, RVA[0:3])))
    
    D3 = np.linalg.det(np.vstack((L[1], RVA[6:9], Ldot2)))
    D4 = np.linalg.det(np.vstack((L[1], RVA[0:3], Ldot2)))

    # Computing all the needed coeficients to find r
    c8 = 1
    c6 = ((4*C*D1)/D - (4*D1**2)/D**2 - R**2)
    c3 = mu*((4*C*D2)/D - (8*D1*D2)/D**2)
    c0 = -(4*mu**2*D2**2)/D**2

    # We normalize the polynomial to earn accuracy, then we find the roots
    coefs = np.array([c8, 0, c6, 0, 0, c3, 0, 0, c0])
    coef_normed = np.array(coefs)/np.abs(coefs).max()
    Z = np.roots(coef_normed)
    
    # print(D,D1,D2,D3,D4)
    # We look for all real and positive roots.
    RV = []
    for each in Z:
        if np.isreal(each) and each > 0:
            r = np.real(each)
            rho = -2*(D1/D) - 2*(mu/(r**3)) * (D2/D)
            rhodot = -1*(D3/D) - 1*(mu)/(r**3) * (D4/D)
            r2 = rho*L[1] + RVA[0:3]
            v2 = rhodot*L[1] + rho*Ldot + RVA[3:6]
            RV.append( np.hstack([r2,v2 ] ) )
    # RV = np.reshape(RV, (-1, 6))
    RV = np.vstack(RV)

    # Then select the best one by comparing to the extra observation data
    Beta = math.pi
    
    P=np.zeros((RV.shape[0],len(Tchecks)))
    for i,row in enumerate(RV):
        # propagate_rv(delta_t, r10, v10, mu, step = 100)
        # propagate_rv(t10, r10, v10, tf, mu)
        
        for k in range(len(Tchecks)):
            check = propagators.propagate_rv(T[1], row[0:3], row[3:6], Tchecks[k],mu)
            
            check = check[-1]
            r_check = check[0:3]
            L_r_check = r_check - r_site
            L_r_check = L_r_check/nplinalg.norm(L_r_check)
            # print("check = ",check)
            # chnrm = check[0:3]/nplinalg.norm(check[0:3])
            angle = math.acos(np.dot(Lchecks[k], L_r_check))
            # print("angle = ",angle,angle*180/np.pi)
            if angle < angleThres:
                P[i,k]=1
    # print("P=",P)
    # print("RV=",RV)
    Nv = len(Tchecks)
    RV = RV[np.sum(P,axis=1)>ThresVotefrac*Nv,:]
    return RV





'''obs_data = radar_filter.radar_filter()

step = 5
actual_pos = obs_data[0:25, 1:4]
obs = np.array([-23.02331*tl.to_rad(), -67.75379*tl.to_rad(), 6378])
r_site = tl.LongLatHeigth_to_cart(obs)
L = np.empty((4,3))

for i in range(4):
    #rot_angle = i * step * omega_e[2]
    #R = np.array([[np.cos(rot_angle), -np.sin(rot_angle), 0], [np.sin(rot_angle), np.cos(rot_angle), 0], [0, 0, 1]])   REPLACE IF EARTH ROTATION
    L[i, :] = actual_pos[i, :] - r_site #- (np.dot(R, r_site))   REPLACE IF EARTH ROTATION
    L[i, :] = L[i, :]/np.linalg.norm(L[i, :])
print(L)
T = np.array([0, step, 2*step, 3*step, 4*step])
#rot_angle = 1 * step * omega_e[2]
#R = np.array([[np.cos(rot_angle), -np.sin(rot_angle), 0], [np.sin(rot_angle), np.cos(rot_angle), 0], [0, 0, 1]])   REPLACE IF EARTH ROTATION
print(r_site)
rv = laplace_orbit_fit(r_site, T, L) #rv = laplace_orbit_fit(np.dot(R, r_site), T, L)  REPLACE IF EARTH ROTATION
r = rv[0:3]
v = rv[3:6]
Orbit1 = tl.from_RV_to_OE(r, v)
print(Orbit1)
my_path = 'C:\\Users\\Thibault\\Desktop\\UAH2021\\Master Thesis\\Data'
my_file = '100_brightest.csv'
objs = pd.read_csv(os.path.join(my_path, my_file))
oe = tl.get_OE_from_TLE(objs)
print(oe.iloc[5:7])
pd.set_option('display.max_columns', None)
pos, vel = tl.from_OE_to_RV(oe.iloc[5])
tru_pos = propagate(5000, pos, vel, step=50)
calc_pos = propagate(5000, r, v, step=50)
fig = plt.figure()
ax1 = plt.gca(projection='3d')
ax1.plot3D(tru_pos[:, 0], tru_pos[:, 1], tru_pos[:, 2], 'red')
ax1.plot3D(calc_pos[:, 0], calc_pos[:, 1], calc_pos[:, 2], 'blue')
#plt.show()
'''








