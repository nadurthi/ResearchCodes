#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:22:17 2019

@author: na0043
"""

import numpy as np
from enum import Enum, auto
import collections as clc


PlanetConstants=clc.namedtuple('PlanetConstants', 'radius radii g mu')
Earth = PlanetConstants(6378.137, [6378.137, 6378.137, 6378.137], 9.8065, 398600) 




        