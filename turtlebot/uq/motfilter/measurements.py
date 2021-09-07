#!/usr/bin/env python
"""
Documentation for this imm module

More details.
"""

import logging
import numpy as np
import uuid


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from uq.uqutils import recorder


class MeasurementScan:
    """
    It can have one measurement or multiple measurements from each sensor
        - ID
        - Meta data
        - time step
        - Scan set (GPS,IMU,IMAGES)
        - Processed data --> bounding box
        - Environment data
        - Context
        - also record the sensor ID (use the sensor model)

    rawmeas: is a dict of scans for each sensor
        : {'SensorID1': [zk1,zk2]
           'SensorID2': [zk1,zk2]
           }
    process the raw measurements and place the in zk
    self.zk: is also a dict
        : {'SensorID1': [zk1,zk2,zk3,zk4]
           'SensorID2': [zk1,zk2,zk3,zk4,zk5]
           }
        you can have more processed measurements
    """

    def __init__(self,t,rawmeas=[],recordit = False):
        self.ID = uuid.uuid4()
        self.t = t
        self.rawmeas = rawmeas
        self.recorder = recorder.StatesRecorder_list(statetypes = {'t':(None,),'rawmeas':(None,),'rawmeas':(None,)} )
        self.recordit = recordit


        # derived measurements
        self.zk = []
        self.association_matrix = []



    def processRawMeas(self,targetset=None,sensorset=None,mapobj=None):
        self.zk = []
        self.association_matrix = []


class XYMeasScan(MeasurementScan):
    def __init__(self,sensorID,tk,rth=[]):
        super().__init__()
        self.tk = tk
        self.meas = rth
        self.sensorID = sensorID
        self.hn = 2

class XYZMeasScan(MeasurementScan):
    def __init__(self,sensorID,tk,rth=[]):
        super().__init__()
        self.tk = tk
        self.meas = rth
        self.states = ['x','y','z']
        self.sensorID = sensorID
        self.hn = 2


class RadialBearingMeasScan(MeasurementScan):
    def __init__(self,sensorID,tk,rth=[]):
        super().__init__()
        self.tk = tk
        self.meas = rth
        self.states = ['r','th']
        self.sensorID = sensorID
        self.hn = 2

class RadialMeasScan(MeasurementScan):
    def __init__(self,sensorID,tk,r=[]):
        super().__init__()
        self.tk = tk
        self.meas = r
        self.states = ['r']
        self.sensorID = sensorID
        self.hn = 1

class BearingMeasScan(MeasurementScan):
    def __init__(self,sensorID,tk,th=[]):
        super().__init__()
        self.tk = tk
        self.meas = th
        self.states = ['th']
        self.sensorID = sensorID
        self.hn = 1

class MonocularImgScan(MeasurementScan):
    def __init__(self,sensorID,tk,imgs=[]):
        super().__init__()
        self.tk = tk
        self.meas = imgs
        self.states = ['img']
        self.sensorID = sensorID
        self.derivedmeas = None
        self.hn = 1

class StereoImgScan(MeasurementScan):
    def __init__(self,sensorID,tk,imgpair=[]):
        super().__init__()
        self.tk = tk
        self.meas = imgpair
        self.states = ['imgL','imgR']
        self.sensorID = sensorID
        self.derivedmeas = None
        self.hn = 1

class PointCloudScan(MeasurementScan):
    def __init__(self,sensorID,tk,scan=[]):
        super().__init__()
        self.tk = tk
        self.meas = scan
        self.states = ['x','y','z','I']
        self.sensorID = sensorID
        self.derivedmeas = None
        self.hn = 1

class IMUScan(MeasurementScan):
    def __init__(self,sensorID,tk,imu=[]):
        super().__init__()
        self.tk = tk
        self.meas = imu
        self.states = ['ax','ay','az','wx','wy','wz']
        self.sensorID = sensorID
        self.derivedmeas = None
        self.hn = 1


class GPSScan(MeasurementScan):
    def __init__(self,sensorID,tk,gps=[]):
        super().__init__()
        self.tk = tk
        self.meas = gps
        self.states = ['x','y','z','lat','lon','height']
        self.sensorID = sensorID
        self.derivedmeas = None
        self.hn = 1



#%% ---------------------------------------------------------------------
class MultipleMeasurementScans:
    """
    At a fixed time step tk
    We can measurements-sets from multiple sources
    """
    def __init__(self):
        self.ID = uuid.uuid4()
        self.tk = 0
        self.measScans = []
