# %% logging
import loggerconfig as logconf
logger = logconf.getLogger(__name__)

logger.info('Info log message')
logger.debug('debug message')
logger.error('error example')
logger.verbose('verbose log message')
logger.warning('warn message')
logger.critical('critical message')
logger.timing('timing message',{'funcName':"funny",'funcArgstr':"None",'timeTaken':3.33})

# %% imports
import os
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import numpy as np
import numpy.linalg as nplg
from uq.gmm import gmmbase as uqgmmbase
from uq.gmm import merger as uqgmmmerg
from uq.gmm import splitter as uqgmmsplit
from utils.math import geometry as utmthgeom
from uq.uqutils import simmanager as uqsimmanager
from utils.plotting import geometryshapes as utpltshp
from utils.plotting import surface as utpltsurf
plt.close('all')
from sklearn import mixture
import time
import utils.timers as utltm





# %% file-level properties

runfilename = __file__
metalog="""
AAS 2020 MOC vs GMM paper simulations
Author: Venkat
Date: June 4 2020

Comparing the IMM+JPDA filter using EKF and UT and CUT points
"""


simmanger = uqsimmanager.SimManager(t0=0,tf=55,dt=0.5,dtplot=0.1,
                                  simname="AIAA-SCITECH-2020-JPDA-IMM",savepath="simulations",
                                  workdir=os.getcwd())

simmanger.initialize()


# %% Create Map






# %% Set UAVs



# %% Create Targets

# search map using multiple Gaussians: search-track





# %% Run sim









# %% Finalize and save

# simmanger.finalize()

# simmanger.save(metalog, mainfile=runfilename, jpdametrics=jpdametrics, jpdamotlist=jpdamotlist)


