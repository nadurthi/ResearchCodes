#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 11:24:14 2021

@author: na0043
"""


def func(x,y):
    z = x + y
    return z

class FuncClass:
    def __init__(self):
        self.g=30
    
    def setg(self,g):
        self.g = g
        
    
    def add(self,x,y):
        z= x + y
        return z
    
    def specialadd(self,x,y):
        z = self.g*x+y
        return z

#%% 

if __name__=="__main__":
    
    print("func(x,y) = ",func(3,2))
    
    fcls1 = FuncClass()
    fcls1.setg(1)
    
    fcls2 = FuncClass()
    fcls2.setg(2)
    
    print("fcls1(x,y) = ",fcls1.specialadd(3,2))
    print("fcls2(x,y) = ",fcls2.specialadd(3,2))
    
