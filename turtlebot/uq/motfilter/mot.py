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

from physmodels import targets as phytarg
from uq.uqutils import recorder as uqrecorder

from uq.motfilter import measurements
from physmodels.sensormodels import SensorSet

# def foo():
#     logger.debug('depending on configuration this may not be printed')
#     logger.info('Log message from function foo.')


class MOTfilter:
    """
    multi-targets multi-measurments processing tracking
    """

    def __init__(self, filterer, recordMeasurementSets=True,recorderobjmeas=None):
        self.ID = uuid.uuid4()
        self.targetset = phytarg.TargetSet()
        self.sensorset = SensorSet()
        self.filterer = filterer
        self.currt = 0
        self.recordMeasurementSets = recordMeasurementSets


        if recorderobjmeas is None:
            self.recordermeas = uqrecorder.StatesRecorder_list(statetypes = ['Zk'] )
        else:
            self.recordermeas = recorderobjmeas




    def propagate(self,t, dt,  filterer,Uk=None, **kwargs):
        self.currtk = self.currtk + 1
        self.filterstage = 'Time Updated'
        if Uk is None:
            Uk = [None]*self.targetset.ntargs


        for i in range(self.targetset.ntargs):
            if self.targetset[i].isactive():
                pass

    def measUpdate(self, Zk, *args, **kwargs):
        """
        Zk is an instance of MeasurementSet

        """
        # if not isinstance(Zk,measurements.MeasurementSet):
        #     raise TypeError("Zk has to motutils.MeasurementSet")



        for i in range(self.targetset.ntargs):
            if self.targetset[i].isactive():
                pass


