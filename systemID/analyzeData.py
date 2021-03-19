import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.5f}'.format


folder = "set (better sync)/"
dopt1 = pd.read_csv(folder+"optitrackData.csv",names=['t','x','y','z','qx','qy','qz','qw'])
dopt=dopt1[dopt1['t']>0]


data = np.load(folder+'IMUdata.npz')
dimu11 = pd.DataFrame(data['X1'],columns=['t','ax','ay','az','wx','wy','wz']) 
dimu22  = pd.DataFrame(data['X2'],columns=['t','ax','ay','az','wx','wy','wz'])

dimu1 = dimu11[dimu11['t']>0] 
dimu2 = dimu22[dimu22['t']>0]


