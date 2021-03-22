import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import systemID.timedataProcessors as sysIDtdp
import systemID.rigidbody_imu_calib as sysIDrgdimucalib
import numpy.linalg as nplinalg
pd.options.display.float_format = '{:.5f}'.format
from scipy.spatial.transform import Rotation as sctrR
from scipy.optimize import minimize
from scipy import interpolate
from scipy.optimize import least_squares
from scipy.linalg import sqrtm
import scipy.integrate
from scipy.integrate import cumtrapz
import quaternion

plt.close("all")
folder = "systemID/set (better sync)/"
# folder = "systemID/set 2/"
dopt1 = pd.read_csv(folder+"optitrackData.csv",names=['t','x','y','z','qx','qy','qz','qw'])
dopt=dopt1[dopt1['t']>0].copy()
# in opttrack, qw is the scalar
# The corresponding Rotation matrix is nothing but Rgo .... o to g


data = np.load(folder+'IMUdata.npz')
dimu11 = pd.DataFrame(data['X1'],columns=['t','ax','ay','az','wx','wy','wz']) 
dimu22  = pd.DataFrame(data['X2'],columns=['t','ax','ay','az','wx','wy','wz'])


dimu1 = dimu11[dimu11['t']>0].copy()
dimu2 = dimu22[dimu22['t']>0].copy()

rgdcalib = sysIDrgdimucalib.RigidBodyCalibrator(dopt,[dimu1,dimu2])

rgdcalib.initialize()
rgdcalib.get_time_statistics()

rgdcalib.estimate_true_rates_spline(k=3,s=0.002)
rgdcalib.calib_imus(10,110,method='spline')


rgdcalib.plotopt(method='spline')

rgdcalib.plotimu()

rgdcalib.plot_imu_opt_with_calib(method='spline')


#%% 





