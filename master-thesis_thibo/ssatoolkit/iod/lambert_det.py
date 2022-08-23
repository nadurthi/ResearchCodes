import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.optimize

mu = 398600 # km3/s2


def lambert_eq(a,*args):
    r1, r2, t1, t2, theta = args
    c = math.sqrt(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * math.cos(theta))
    s = 1 / 2 * (r1 + r2 + c)
    alpha = math.acos(1-s/a)
    beta = math.acos(1-(s-c)/a)

    return a**1.5 * ((alpha - math.sin(alpha)) - (beta - math.sin(beta))) - math.sqrt(mu)*(t2-t1)
r1 = np.array([6045, 3490, 0])
r2 = np.array([12214.839, 10249.467, 2000])
theta = 1
data = (np.linalg.norm(r1), np.linalg.norm(r2), 0, 1092, theta*math.pi/180)
a = scipy.optimize.fsolve(lambert_eq, 12000, args=data)
a = a[0]
u1 = r1/np.linalg.norm(r1)
u2 = r2/np.linalg.norm(r2)
c = math.sqrt(np.linalg.norm(r1) ** 2 + np.linalg.norm(r2) ** 2 - 2 * np.linalg.norm(r1) * np.linalg.norm(r2) * math.cos(theta*math.pi/180))
s = 1 / 2 * (np.linalg.norm(r1) + np.linalg.norm(r2) + c)
alpha = math.acos(1-s/a)
beta = math.acos(1-(s-c)/a)
uc = (r2-r1)/c
print(a)
A = math.sqrt(mu/(4*a))/math.tan(alpha/2)
B = math.sqrt(mu/(4*a))/math.tan(beta/2)

v1 = (B+A)*uc + (B-A)*u1
v2 = (B+A)*uc - (B-A)*u2

print(v1)
print(v2)

