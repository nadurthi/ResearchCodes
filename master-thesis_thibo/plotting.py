#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 11:12:49 2022

@author: na0043
"""

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import ssatoolkit.propagators as ssaprop
# import toolkit as tl
import math
import numpy as np
import pandas as pd
from numpy import linalg as nplinalg

def plotPlanet(ax,planet):
    r = planet.radius
    u = np.linspace(0, 2 * np.pi, 100)
    uu = np.linspace(0, np.pi, 100)
    x = r * np.outer(np.cos(u), np.sin(uu))
    y = r * np.outer(np.sin(u), np.sin(uu))
    z = r * np.outer(np.ones(np.size(u)), np.cos(uu))
    ax.plot_surface(x, y, z, color='b')

    
def plot_data(true_trajectories, orb_elem,t0, tf, dt,):
    #plot the earth
    r = 6378
    fig = plt.figure()
    ax1 = plt.gca(projection='3d')
    u = np.linspace(0, 2 * np.pi, 100)
    uu = np.linspace(0, np.pi, 100)
    x = r * np.outer(np.cos(u), np.sin(uu))
    y = r * np.outer(np.sin(u), np.sin(uu))
    z = r * np.outer(np.ones(np.size(u)), np.cos(uu))
    ax1.plot_surface(x, y, z, color='b')

    # We plot the true trajectories
    #print(true_trajectories)
    for key in true_trajectories:
        #print(true_trajectories[key][:, 0])
        ax1.plot3D(true_trajectories[key][:, 0], true_trajectories[key][:, 1], true_trajectories[key][:, 2], 'black')

    # And the estimated trajectories
    for each_sat in orb_elem:
        traj_data = ssaprop.propagate_kep(t0, tf, dt, each_sat)
        ax1.plot3D(traj_data[0], traj_data[1], traj_data[2], 'red')
    # Now, that we have everything, we launch the plot
    plt.show()


def draw_fov(a, b, obs_cart):
    a_par_b = (np.dot(a, b)/np.dot(b, b))*b
    a_per_b = a - a_par_b
    w = np.cross(b, a_per_b)
    vec = np.linspace(0, 2*math.pi, 200)
    vision = np.array([])
    for theta in vec:
        x1 = math.cos(theta)/np.linalg.norm(a_per_b)
        x2 = math.sin(theta)/np.linalg.norm(w)
        a_per_b_theta = np.linalg.norm(a_per_b)*(x1*a_per_b + x2*w)
        a_theta = a_per_b_theta + a_par_b
        fov = obs_cart + a_theta
        vision = np.append(vision, fov)
    vision = np.reshape(vision, (-1,3))
    return vision




def plot_cone(ax,rsite,Lorient,DepthMax,half_angle,plotparams):
    """
    rsite is location of cone center
    Loreint is unit vector along the cone cneter (pointing direction)
    half_angle is the half_angle of the cone.
    DepthMax is the max length of cone along the center
    """
    # max radius is
    
    

        
    # Set up the grid in polar
    Rmax =  DepthMax*np.tan(half_angle)
    theta = np.linspace(0,2*np.pi,50)
    r = np.linspace(0,Rmax,10)
    T, R = np.meshgrid(theta, r)
    
    # Then calculate X, Y, and Z
    X = R * np.cos(T)
    Y = R * np.sin(T)
    # Z = np.zeros_like(X)
    Z = np.sqrt(X**2 + Y**2)/np.tan(half_angle)
    # for i in range(X.shape[0]):
    #     for j in range(X.shape[1]):
    #         Z[i,j] = np.sqrt(X[i,j]**2 + Y[i,j]**2)/np.tan(half_angle)
    
    
    Lorient=Lorient/nplinalg.norm(Lorient)
    rot_angle = np.arccos(np.dot(Lorient,[0,0,1]))
    rot_axis = np.cross([0,0,1],Lorient)
    rot_axis_mat = np.array([[0,-rot_axis[2],rot_axis[1]],
                             [rot_axis[2],0,-rot_axis[0]],
                             [-rot_axis[1],rot_axis[0],0]])
    # Rodriguez idenity for rotation.
    R = np.identity(3)+np.sin(rot_angle)*rot_axis_mat+(1-np.cos(rot_angle))*np.matmul(rot_axis_mat,rot_axis_mat)
    
    # rotate and translate the cone to the desired 
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i,j],Y[i,j],Z[i,j] = R.dot([ X[i,j],Y[i,j],Z[i,j] ])+rsite
            
    
    # Set the Z values outside your range to NaNs so they aren't plotted
    # ax.plot_wireframe(X, Y, Z)
    # plotparams={'color':'b','alpha':0.3, 
    #                                  'linewidth':0, 
    #                                  'antialiased':False}
    ax.plot_surface(X, Y, Z, **plotparams)
    
    return ax



