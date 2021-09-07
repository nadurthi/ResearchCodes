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

class PDA(mot.MOTfilter):
    def __init__(self, recordMeasurementSets=True):

        super().__init__(recordMeasurementSets=recordMeasurementSets)
