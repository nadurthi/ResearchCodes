#!/usr/bin/env python
"""
Documentation for this imm module

More details.
"""


import mot
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# %% usage
# tgt = Target(dynModels=[],xf0=xf0,Pf0=Pf0,recordfilterstate=True,status='active')
# PDA.targetset.addTarget(tgt)

# %%


class NNDA(mot.MOTfilter):
    """
    No data association, All targets get all the measurements
    This is for testing puposes
    - measurements are (x,y)
    """
    def __init__(self, filterer,recordMeasurementSets=True):
        self.dynmodelIdx = 0
        self.gmmIdx = 0
        self.sensorModelIdx = 0
        super().__init__( filterer,recordMeasurementSets=recordMeasurementSets)

    def propagate(self,t, dt, uk, filterer, **kwargs):
        self.currtk = self.currtk + 1
        self.filterstage = 'Time Updated'
        if Uk is None:
            Uk = [None]*self.targetset.ntargs


        for i in range(self.targetset.ntargs):
            if self.targetset[i].isactive():
                # self.targetset[i].propagate(tk,dtk,Uk[i])
                self.filterer.propagate(t,dt,self.targetset[i],Uk[i],
                                        dynmodelIdx=self.dynmodelIdx,
                                        gmmIdx = self.gmmIdx,
                                        updttarget=True)


    def measUpdt(self, Zk, *args, **kwargs):
        """
        Zk is an instance of MeasurementSet

        """
        if not isinstance(Zk,measurements.MeasurementSet):
            raise TypeError("Zk has to motutils.MeasurementSet")

        self.filterstage = 'Measurement Updated'

        for i in range(self.targetset.ntargs):
            if self.targetset[i].isactive():
                filterer.measUpdt(t,dt,self.targetset[i],
                                sensModel=self.sensorset[self.sensorModelIdx],
                                dynmodelIdx=self.dynmodelIdx,
                                gmmIdx=0,updttarget=True)
