import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import toolkit as tl
from prop_odeint import propagate
import sys

def radar_filter():
    my_path = 'C:\\Users\\Thibault\\Desktop\\UAH2021\\Master Thesis\\Data'
    my_file = '100_brightest.csv'
    obs = np.array([[-23.02331, -67.75379, 6378],   # Attacama Observatory
                   [49.133949, 1.441930, 6378],        # Pressagny l'Orgeuilleux Observatory
                   [35.281287, -116.783051, 6378]])
     #   [0,0,6378])
    obs_angle = 25 # Half angle of observation (FOV=2*obs_angle)
    range_obs = 2000 # Max range of observation of the observatory
    mu = 398600 # km3/s2
    r = 6378 # km


    if obs.ndim == 1:
        theta = obs[0] * tl.to_rad()
        phi = obs[1] * tl.to_rad()
        r = obs[2]
        obs_cart = tl.LongLatHeigth_to_cart(obs)
        r_max = np.array([(r+range_obs) * np.cos(theta) * np.cos(phi),
                                    (r+range_obs) * np.cos(theta) * np.sin(phi),
                                    (r+range_obs) * np.sin(theta)])
        obs_vect = r_max - obs_cart

    if obs.ndim == 2:
        obs_cart = np.zeros((obs.shape[0], 3))
        r_max = np.zeros((obs.shape[0], 3))
        i = 0
        for row in obs:
            theta = row[0]*tl.to_rad()
            phi = row[1]*tl.to_rad()
            r = row[2]
            obs_cart[i, :] = tl.LongLatHeigth_to_cart(row)
            r_max[i, :] = [(r+range_obs) * np.cos(theta) * np.cos(phi),
                                    (r+range_obs) * np.cos(theta) * np.sin(phi),
                                    (r+range_obs) * np.sin(theta)]
            obs_vect = r_max - obs_cart
            i = i + 1
    #print(obs_cart)
    #print(r_max)
    #print(obs_vect)

    objs = pd.read_csv(os.path.join(my_path, my_file))

    # Plot earth
    fig = plt.figure()
    ax1 = plt.gca(projection='3d')
    u = np.linspace(0, 2 * np.pi, 100)
    uu = np.linspace(0, np.pi, 100)
    x = r * np.outer(np.cos(u), np.sin(uu))
    y = r * np.outer(np.sin(u), np.sin(uu))
    z = r * np.outer(np.ones(np.size(u)), np.cos(uu))
    #ax1.plot_surface(x, y, z, color='b')




    observed_sat = np.array([])


    for index, row in objs.iterrows():
        ok = False
        E = tl.Find_Kep_E(row["ECCENTRICITY"], row["MEAN_ANOMALY"]*math.pi/180)
        f = np.arccos((np.cos(E) - row["ECCENTRICITY"]) / (1 - row["ECCENTRICITY"] * np.cos(E)))

        if (int(E / math.pi) % 2) == 1:
            f = 2 * math.pi - f
        a = (mu/(2*math.pi*row["MEAN_MOTION"]/86400)**2)**(1/3)
        e = row["ECCENTRICITY"]
        i = row["INCLINATION"]*math.pi/180
        Om = row["RA_OF_ASC_NODE"]*math.pi/180
        om = row["ARG_OF_PERICENTER"] * math.pi / 180
        f = f*math.pi/180
        orbital_elements = np.array([a, e, i, Om, om, f])

        r, v = tl.from_OE_to_RV(orbital_elements)
        orb_time = 2*math.pi*math.sqrt(orbital_elements[0]**3/mu)
        states = propagate(orb_time, r, v, step=5)

        in_fov_x = np.array([])
        in_fov_y = np.array([])
        in_fov_z = np.array([])
        ok = False
        for each in states:
                if obs.ndim == 1:
                    pos_to_obs = np.array(
                        ([each[0] - obs_cart[0], each[1] - obs_cart[1], each[2] - obs_cart[2]]))
 #                   print(each)
 #                   print(pos_to_obs)
                    Beta = math.acos((pos_to_obs.dot(obs_vect)) / (
                                np.linalg.norm(pos_to_obs) * np.linalg.norm(obs_vect)))
                if obs.ndim == 2:
                    pos_to_obs = np.array([])
                    Beta = np.array([])
                    for x in range(obs.shape[0]):
                        pos_to_obs = np.append(pos_to_obs, [each[0]-obs_cart[x, 0], each[1]-obs_cart[x, 1], each[2]-obs_cart[x, 2]])
                        pos_to_obs = np.reshape(pos_to_obs, (-1, 3))

                        Beta = np.append(Beta, math.acos((pos_to_obs[-1, :].dot(obs_vect[x, :]))/(np.linalg.norm(pos_to_obs[-1, :]) * np.linalg.norm(obs_vect[x, :]))))

                    observatory_ID = np.where(Beta == np.amin(Beta))[0]

                if abs(Beta[observatory_ID]) < obs_angle*tl.to_rad():
                    xdata = states[:, 0]
                    ydata = states[:, 1]
                    zdata = states[:, 2]
                    in_fov_x = np.append(in_fov_x, each[0])
                    in_fov_y = np.append(in_fov_y, each[1])
                    in_fov_z = np.append(in_fov_z, each[2])
                    ok = True
                    observed_sat = np.concatenate([observed_sat, np.array([index, each[0], each[1], each[2], int(observatory_ID)])])





        for each in states:
            radius = math.sqrt(each[0]**2 + each[1]**2 + each[2]**2)
            if radius < 6378:
                print("SSSSTTTTOOOOPPPP")
        if ok == True:
            ax1.plot3D(xdata, ydata, zdata,'red')
            ax1.plot3D(in_fov_x, in_fov_y, in_fov_z, 'green')
    #plt.show()
    observed_sat = observed_sat.reshape(-1, 5)
    return observed_sat
'''
result = radar_filter()
#print(result.shape)
np.set_printoptions(threshold=sys.maxsize)
#print(result)
np.savetxt("obs_data1.csv", result,delimiter=',')
'''


