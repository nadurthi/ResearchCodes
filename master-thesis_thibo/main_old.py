import scipy
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def Find_Kep_E(e,M):
    i = 0
    tol = 1E-12
    B = np.cos(e) - ((math.pi/2) - e)*np.sin(e)
    E = np.array([])
    fE = np.array([])
    dfE = np.array([])
    d2fE = np.array([])
    E1 = np.array([])
    E2 = np.array([])
    E = np.append(E, M + (e*np.sin(M))/(B+M*np.sin(e)))
    while np.abs(E[i] - M - e*np.sin(E[i])) > tol:

        fE = np.append(fE, E[i] - e*np.sin(E[i]) - M)
        dfE = np.append(dfE,  1 - e*np.cos(E[i]))
        d2fE = np.append(d2fE, e*np.sin(E[i]))

        A = 2*np.sqrt((np.abs(4*(dfE[i]**2))))
        E1 = np.append(E1, E[i] - (5*fE[i]/(dfE[i] + A)))
        E2 = np.append(E2, E[i] - (5*fE[i]/(dfE[i] - A)))

        if (abs(E1[i] - E[i])) < (abs(E2[i] - E[i])):
            E = np.append(E, E1[i])
        else:
                E = np.append(E, E2[i])
        i = i + 1

    if i > 100:
        print(" Kepler's Equation is NOT converging ! ")
        return
    return E[i]

def pqwTOijk(A, Om, om, i):
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


def simple_prop(r, v, Delta_t, step=100):

    prop_r = np.empty((0,3))
    prop_v = np.empty((0,3))
    mu = 398600  # km3/s2
    Ihat = np.array([1, 0, 0])
    Jhat = np.array([0, 1, 0])
    Khat = np.array([0, 0, 1])

    h = np.cross(r, v) #Angular momentum vector
    e = np.cross(v, h)/mu - r/np.linalg.norm(r) #Eccentricity vector
    n = np.cross(Khat, h) #Node vector

    # Let's now find i, Omega, omega and the vu. They are the orbital elements of our initial orbit
    i = np.arccos(h.dot(Khat)/np.linalg.norm(h)) # Inclination
    Om = np.arccos(n.dot(Ihat)/np.linalg.norm(n)) # Ascending node
    om = np.arccos(n.dot(e)/(np.linalg.norm(n)*np.linalg.norm(e))) # periapsis position
    f = np.arccos(e.dot(r)/(np.linalg.norm(e)*np.linalg.norm(r))) # True anomaly
    a = np.linalg.norm(r)/(2-((np.linalg.norm(r)*np.linalg.norm(v)**2)/mu))
    Orbit1 = np.array([a, np.linalg.norm(e), i, Om, om, f])


    # Now that we have all the orbital elements, we must find the future true anomaly of the next step, and all the following steps:
    E = np.arccos((Orbit1[1] + np.cos(Orbit1[5]))/(1 + Orbit1[1] * np.cos(Orbit1[5])))  # Mean eccentricity
    current_Dt = np.sqrt(Orbit1[0]**3/mu)*(E - Orbit1[1] * np.sin(E))  # We compute the delta t of the first observation
    i = 0
    while current_Dt + Delta_t > current_Dt + step * i:
        M = math.sqrt(mu/Orbit1[0]**3) * (current_Dt + step * i)  # We compute the mean anomaly at the desired future position

        E = Find_Kep_E(Orbit1[1], M)
        Orbit1[5] = np.arccos((np.cos(E) - Orbit1[1])/(1 - Orbit1[1] * np.cos(E)))

        if (int(E/math.pi) % 2) == 1:
            Orbit1[5] = 2 * math.pi - Orbit1[5]


        # And finally, find the new r and v:
        p = Orbit1[0]*(1-Orbit1[1]**2)

        r[0] = p*np.cos(Orbit1[5])/(1+Orbit1[1]*np.cos(Orbit1[5])) # Here we compute the coordinates in the PQW coordinate system
        r[1] = p*np.sin(Orbit1[5])/(1+Orbit1[1]*np.cos(Orbit1[5]))
        r[2] = 0
        v[0] = -np.sqrt(mu/p)*np.sin(Orbit1[5])
        v[1] = np.sqrt(mu/p)*(Orbit1[1]+np.cos(Orbit1[5]))
        v[2] = 0

        prop_r = np.append(prop_r, pqwTOijk(r, -Orbit1[3], -Orbit1[4], -Orbit1[2])) # We Convert the PQW coordinates to XYZ coordinates
        prop_r = np.reshape(prop_r, (i+1, 3))
        prop_v = np.append(prop_v, pqwTOijk(v, -Orbit1[3], -Orbit1[4], -Orbit1[2]))
        prop_v = np.reshape(prop_v, (i+1, 3))

        i = i + 1
    return prop_r, prop_v

r = np.array([-4069.503, 2861.786, 4483.608]) #Observation position
v = np.array([-5.114, -5.691, -1.000]) #Observation speed
rv = simple_prop(r, v, 28600)
r = rv[0]
v = rv[1]
print(r)

fig = plt.figure()
ax = plt.axes(projection='3d')
xdata = r[:, 0]
ydata = r[:, 1]
zdata = r[:, 2]
ax.plot3D(xdata, ydata, zdata,'green')
plt.show()
