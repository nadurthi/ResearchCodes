#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:22:17 2019

@author: na0043
"""

import numpy as np
from enum import Enum, auto
import collections as clc


class Planet(Enum):
    Earth = auto()
    Sun = auto()



class PlanetConstants:
    """Constants for each planet."""
    
    def __init__(self, planet=Planet.Earth):
        if planet == Planet.Earth:
            self._radii = tuple([6378.137, 6378.137, 6378.137])
            self._mu = 3.986004418e5
            self._R = 6378.137
            self._g = 9.8065
    
    
    def getTrue(self):
        prop  =clc.namedtuple("Properties",["radii","mu",'R',"g"])(self._radii,
                                                                 self._mu,
                                                                 self._R,
                                                                 self._g)
        return prop
        
    
    def getCanonical(self):
        radiican = np.array(self._radii)/self._R
        prop  =clc.namedtuple("Properties",["radii","mu",'R',"g"])(self._radii,
                                                                 1,
                                                                 1,
                                                                 None)
        return prop





        