# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 23:01:07 2019

@author: Nagnanamus
"""

import math
from numpy import linalg as nplinalg
import numpy as np
import sys
import pdb
from ssatoolkit import date_time

class DimensionalConverter:
    def __init__(self,planetconst):
        self.planetconst = planetconst
        
        self.mu = self.planetconst.mu
        self.RU     = self.planetconst.radius
        self.TU     = np.sqrt(self.RU**3 / self.mu);
        self.VU     = self.RU/self.TU
        
        self.trueA2normA=self.TU**2/self.RU
        self.normA2trueA=self.RU/self.TU**2;
        
        self.trueV2normV=self.TU/self.RU
        self.normV2trueV=self.RU/self.TU
        
        self.trueX2normX=1/self.RU
        self.normX2trueX=self.RU
        
        self.trueT2normT=1/self.TU
        self.normT2trueT=self.TU

    def true2can_acc(self,X):
        return self.trueA2normA*X
    
    def true2can_pos(self,X):
        return self.trueX2normX*X
    
    def true2can_vel(self,X):         
        return self.trueV2normV*X
    def true2can_posvel(self,X):
        XX=X.copy()
        if X.ndim==1:
            n=int(X.size/2)
            XX[0:n] = self.trueX2normX*X[0:n]
            XX[n:] = self.trueV2normV*X[n:]
            return XX
        else:
            n=X.shape[0]
            XX[:,0:n] = self.trueX2normX*X[:,0:n]
            XX[:,n:] = self.trueV2normV*X[:,n:]
            return XX
        
    def true2can_time(self,X):
        return self.trueT2normT*X


def RotMat_z(a):
    return np.array([[np.cos(a),np.sin(a),0],
     [-np.sin(a),np.cos(a),0],
     [0,0,1]])

def RotMat_x(a):
    return np.array([[1,0,0],
                     [0,np.cos(a),np.sin(a)],
                     [0,-np.sin(a),np.cos(a)]])
    
def from_RV_to_OE_vec(rv,mu):
    return from_RV_to_OE(rv[:3],rv[3:],mu)

        
def from_RV_to_OE(r,v,mu):
    """ Convert some position and velocity vectors into some orbital elements.

    Parameters
    ----------

    r: numpy.array (3x1)
      Position vector.
    v: numpy.array (3x1)
      Velocity vector.

    Returns
    -------

    Orbit1: numpy.array (6x1)
      The converted orbital elements.

    """
    Ihat = np.array([1, 0, 0])
    Jhat = np.array([0, 1, 0])
    Khat = np.array([0, 0, 1])
    h = np.cross(r, v)  # Angular momentum vector
    # e = np.cross(v, h) / mu - r / np.linalg.norm(r)  # Eccentricity vector
    rmag = np.linalg.norm(r)
    vmag = np.linalg.norm(v)
    e = 1/mu*( (vmag**2-mu/rmag)*r-np.dot(r,v)*v  )   # Eccentricity vector
    n = np.cross(Khat, h)  # Node vector

    # Let's now find i, Omega, omega and the vu. They are the orbital elements of our initial orbit
    i = np.arccos(h.dot(Khat) / np.linalg.norm(h))  # Inclination
    
    Om = np.arccos(np.dot(n,Ihat) / np.linalg.norm(n))  # Ascending node
    # print(Om)
    if n[1]<=0:
        Om = 2*np.pi-Om
       
    om = np.arccos(n.dot(e) / (np.linalg.norm(n) * np.linalg.norm(e)))  # periapsis position
    if e[2]<0:
        om = 2*np.pi-om
        
    f = np.arccos(e.dot(r) / (np.linalg.norm(e) * np.linalg.norm(r)))  # True anomaly
    
    if np.dot(r,v)<=0:
        f=2*np.pi-f
    
    Energy = vmag**2/2-mu/rmag
    a= - mu/(2*Energy)
    # a = np.linalg.norm(r) / (2 - ((np.linalg.norm(r) * np.linalg.norm(v) ** 2) / mu))
    
    Orbit1 = np.array([a, np.linalg.norm(e), i, Om, om, f])
    return Orbit1

def from_OE_to_RV(Orbit1,mu):
    """ Convert some orbital elements into some position and velocity vectors.

    Parameters
    ----------

    Orbit1: numpy.array
      The set of original orbital elements.

    Returns
    -------

    r: numpy.array
      Position vector.
    v: numpy.array
      Velocity vector.

    """
    [a, e, i, Om, om, f] = Orbit1
    p = a * (1 - e ** 2)
    rmag = p/(1+e*np.cos(f))
    rdotmag = np.sqrt(mu/p)*e*np.sin(f)
    rfdotmag  = np.sqrt(mu/p)*(1+e*np.cos(f))
    
    rbar_pqw = np.array([rmag*np.cos(f),rmag*np.sin(f),0])
    vbar_pqw = np.sqrt(mu/p)* np.array([-np.sin(f),(e+np.cos(f)),0])
    
    pqw_R_ijk = np.matmul(np.matmul(RotMat_z(om),RotMat_x(i)),RotMat_z(Om))
    ijk_R_pqw = pqw_R_ijk.T
    
    r = ijk_R_pqw.dot(rbar_pqw)
    v = ijk_R_pqw.dot(vbar_pqw)
    rv = np.hstack([r,v])
    
    # r = np.array([0.0,0.0,0.0])
    # v = np.array([0.0,0.0,0.0])
    # r[0] = p * math.cos(Orbit1[5]) / (
    #             1 + Orbit1[1] * math.cos(Orbit1[5]))  # Here we compute the coordinates in the PQW coordinate system
    # r[1] = p * np.sin(Orbit1[5]) / (1 + Orbit1[1] * np.cos(Orbit1[5]))
    # r[2] = 0
    # v[0] = -np.sqrt(mu / p) * np.sin(Orbit1[5])
    # v[1] = np.sqrt(mu / p) * (Orbit1[1] + np.cos(Orbit1[5]))
    # v[2] = 0
    # r = pqwTOijk(r, Orbit1[3], -Orbit1[4], -Orbit1[2])
    # v = pqwTOijk(v, Orbit1[3], -Orbit1[4], -Orbit1[2])
    # rv = np.concatenate((r, v))
    return rv

def to_rad():
    """ provide the formula to convert an angle in degrees to radians.

    Parameters
    ----------

    NAN.

    Returns
    -------

    to_rad: float
      The converting formula.

    """
    to_rad = math.pi/180
    return to_rad

    
def geodetic_to_ECI(lat, lon, alt, yy, mm, dd, hh, min, ss):
    f_E = 1/298.257223563
    SMA_E = 6378137.0
    OMEGA_E = 7.2921150 * 10**(-5)
    sl = np.sin(lat)
    e2 = f_E * (2 - f_E)
    p = (SMA_E/np.sqrt(1 - e2*sl*sl) + alt) * np.cos(lat)
    q = (SMA_E*(1 - e2)/np.sqrt(1 - e2*sl*sl) + alt) * sl
    lmst = date_time.GMST(yy, mm, dd, hh, min, ss) + lon % 2*math.pi
    ct = np.cos(lmst)
    st = np.sin(lmst)
    ECI_vec = np.array([p*ct, p*st, q])
    ECI_vec = np.concatenate(ECI_vec, np.array([-st, ct, 0])*p*OMEGA_E)
    ECI_vec = np.concatenate(ECI_vec, np.array([-ct, -st, 0])*p*OMEGA_E*OMEGA_E)
    ECI_vec = np.transpose(ECI_vec)
    return ECI_vec





def LongLatHeigth_to_cart(sph_coord):
    """ Convert Longitude, Latitude and Altitude coordinates to Cartesians.

    Parameters
    ----------

    sph_coord: numpy.array (3x1)
      Longitude, latitude and altitude coordinates.

    Returns
    -------

    cart_coord: numpy.array (3x1)
      The asked cartesian coordinates.

    """
    # Spherical coordinates extraction
    theta = sph_coord[0]
    phi = sph_coord[1]
    r = sph_coord[2]

    # Cartesian coordinates computation
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)
    cart_coord = np.array([x, y, z])

    return cart_coord


def pqwTOijk(A, Om, om, i):
    """ Convert a set of coordinates from PQW to IJK system.

    Parameters
    ----------

    A: numpy.array (3x1)
      The set of coordinates in PQW.
    Om: float
      Angle of the ascending node.
    om: float
      Angle to perigee.
    i: float
      Inclination angle.
    Returns
    -------

    res: numpy.array (3x1)
      coordinates in IJK.

    """
    ROTzOm = np.array([[np.cos(Om), np.sin(Om), 0],
                       [-np.sin(Om), np.cos(Om), 0],
                       [0, 0, 1]])
    B = ROTzOm
    ROTxi = np.array([[1, 0, 0],
                      [0, np.cos(i), np.sin(i)],
                      [0, -np.sin(i), np.cos(i)]])
    B = B.dot(ROTxi)
    ROTzom = np.array([[np.cos(om), np.sin(om), 0],
                       [-np.sin(om), np.cos(om), 0],
                       [0, 0, 1]])
    B = B.dot(ROTzom)
    A = B.dot(A.transpose())
    return A


def L_from_cart(actual_pos, obs_pos):
    r_site = LongLatHeigth_to_cart(obs_pos)
    L = actual_pos[0:3] - r_site #- (np.dot(R, r_site))   REPLACE IF EARTH ROTATION
    L = L/np.linalg.norm(L)
    return L

def L_from_cart_rsite(actual_pos, r_site):
    # r_site = LongLatHeigth_to_cart(obs_pos)
    L = actual_pos[0:3] - r_site #- (np.dot(R, r_site))   REPLACE IF EARTH ROTATION
    L = L/np.linalg.norm(L)
    return L

def cart2spherical(x,y,z):
    
    r=np.sqrt(x**2+y**2+z**2)
    # print(np.sqrt(x**2,y**2),z)
    theta = np.arctan2(np.sqrt(x**2+y**2),z)
    phi = atn = np.arctan2(y,x) 
    # if x>0:
    #     phi = atn
    # elif x<0 and y>=0:
    #     phi = atn+np.pi
    # elif x<0 and y<0:
    #     phi = atn-np.pi
    # elif x==0 and y>0:
    #     phi = np.pi/2
    # elif x==0 and y<0:
    #     phi = -np.pi/2
    # elif x==0 and y==0:
    #     phi = np.nan
    
    return np.array([r,theta,phi])
    
def spherical2cart(r,theta,phi):
    x = r*np.cos(phi)*np.sin(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z= r*np.cos(theta)
    return np.array([x,y,z])


# def get_L_from_Pos(pos_data, observatory_pos):
#     r_site = tl.LongLatHeigth_to_cart(observatory_pos)
#     L = np.array([])

#     for i in range(pos_data.shape[0]):
#         L[i, :] = pos_data[i, :] - r_site
#         L[i, :] = L[i, :] / np.linalg.norm(L[i, :])

def get_L_from_Pos(pos_data, observatory_pos, obs_station):
    numb_inter = pos_data.shape[0]
    L = np.empty((numb_inter,3))
    #print(pos_data)
    for i in range(numb_inter):
        r_site = LongLatHeigth_to_cart(observatory_pos[int(obs_station[i]), :])
        L[i, :] = pos_data[i, :] - r_site
        L[i, :] = L[i, :] / np.linalg.norm(L[i, :])
    L = L.reshape(-1, 3)
    print(L[0:4])
    return L

#%%
def cart2classorb(x,mu):
#    x=[x,y,vx,vy]
#    out=[a,e,i,om,Om,M]
    if x.ndim != 1:
        sys.exit('x.ndim has to be 1')
    
    r = x[0:3]
    v = x[3:6] 
    rm = nplinalg.norm(r)
    vm = nplinalg.norm(v)
    a = 1/(2/rm-vm**2/mu)
    
    hbar = np.cross(r,v)
    cbar = np.cross(v,hbar)-mu*r/rm
    
    e=nplinalg.norm(cbar)/mu
    
    hm = nplinalg.norm(hbar)
    ih = hbar/hm;
    
    ie = cbar/(mu*e)
    
    ip = np.cross(ih,ie)
    
    i = np.arccos(ih[2])

#    pdb.set_trace()
    w = np.arctan2(ie[2],ip[2])
    
    Om = np.arctan2(ih[0],-ih[1])
    
    f = np.arccos(np.dot(ip,r)/rm)

    if np.dot(v,r)<=0:
        f=2*np.pi-f
    
    sig = np.dot(r,v)/np.sqrt(mu)
    
    E = np.arctan2(sig/np.sqrt(a),1-rm/a)
    
    M = E-e*np.sin(E)
    
    classorb = np.array([a,e,i,w,Om,M])

    return classorb




    
     