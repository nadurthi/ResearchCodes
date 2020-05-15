#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:22:17 2019

@author: na0043
"""

import numpy as np


constants={}
constants['radii']=np.array([6378.137,6378.137,6378.137]);

constants['mu']      = 3.986004418e5    # % Gravitational Const
constants['Re']      = constants['radii'][0] #          % Earth radius (km)

constants['g']   = 9.8065;            #% Sea-level acceleration, (m/s^2)

#% Canonical Units
constants['muCan']   = 1;
constants['RU']      = constants['Re'];
constants['TU']      = np.sqrt(constants['RU']**3 / constants['mu']);
constants['VU']      = constants['RU']/constants['TU'];

constants['trueA2normA']=constants['TU']**2/constants['RU'];
constants['normA2trueA']=constants['RU']/constants['TU']**2;

constants['trueV2normV']=constants['TU']/constants['RU'];
constants['normV2trueV']=constants['RU']/constants['TU'];

constants['trueX2normX']=1/constants['RU'];
constants['normX2trueX']=constants['RU'];

constants['trueT2normT']=1/constants['TU'];
constants['normT2trueT']=constants['TU'];




class TBPconstants:
    states={}
    states['t'] = 'true states'
    states['c'] = 'canonical states'
    
    def __init__(self):
        self._state = 't'
    
    def canonical(self):
        self._state = 'c'
        return self
    def true(self):
        self._state = 't'
        return self
    
    @property
    def mu(self):
        if self._state == 't':
            return constants['mu']
        elif self._state == 'c':
            return constants['muCan']
    
    @property
    def R(self):
        if self._state == 't':
            return constants['Re']
        elif self._state == 'c':
            return constants['RU']
        
    @property
    def TU(self):
        return constants['TU']
    @property
    def VU(self):
        return constants['VU']
    
    
    def state(self):
        print(self.states[self._state])
        
    @staticmethod
    def trueA2normA(self,a):
        return a*constants['trueA2normA']
    
    @staticmethod
    def normA2trueA(self,a):
        return a*constants['normA2trueA']
    
    @staticmethod
    def trueV2normV(self,v):
        return v*constants['trueV2normV']
    
    @staticmethod
    def normV2trueV(self,v):
        return v*constants['normV2trueV']

    @staticmethod
    def trueX2normX(self,x):
        return x*constants['trueX2normX']
    
    @staticmethod
    def normX2trueX(self,x):
        return x*constants['normX2trueX']
    
    
    @staticmethod
    def trueT2normT(self,t):
        return t*constants['trueT2normT']
    
    @staticmethod
    def normT2trueT(self,t):
        return t*constants['normT2trueT']
    



        