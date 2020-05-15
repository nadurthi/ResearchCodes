#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:27:13 2019

@author: na0043
"""
from . import constants
import numpy as np



class Time:
    _t0=0
    _tf=0
    _dt=0
    _Tvec=0
    Nsteps=0
    _dtplot=0
    _Tvecplot=0
    states={}
    states['t'] = 'true states'
    states['c'] = 'canonical states'
    def __init__(self,t0,dt,tf,dtplot):
        self._state = 't'
        
        self._t0=t0
        self._dt=dt
        self._tf=tf
        
        self._Tvec = np.arange(t0,tf+dt,dt)
        self.Nsteps = len(self.Tvec)
        self._Tvecplot = np.arange(t0,tf+dtplot,dtplot)
        
    def canonical(self):
        self._state = 'c'
        return self
    def true(self):
        self._state = 't'
        return self
    
    
    
    @property
    def t0(self):
        if self._state is 't':
            return self._t0
        elif self._state is 'c':
            return self._t0/constants.constants['TU']
        
    @property
    def dt(self):
        if self._state is 't':
            return self._dt
        elif self._state is 'c':
            return self._dt/constants.constants['TU']
    
    @property
    def dtplot(self):
        if self._state is 't':
            return self._dtplot
        elif self._state is 'c':
            return self._dtplot/constants.constants['TU']
        
        
    @property
    def tf(self):
        if self._state is 't':
            return self._tf
        elif self._state is 'c':
            return self._tf/constants.constants['TU']
        
    @property
    def Tvec(self):
        if self._state is 't':
            return self._Tvec
        elif self._state is 'c':
            return self._Tvec/constants.constants['TU']
    
    @property
    def Tvecplot(self):
        if self._state is 't':
            return self._Tvecplot
        elif self._state is 'c':
            return self._Tvecplot/constants.constants['TU']    



class SimModel:
    def __init__(self):
        pass


