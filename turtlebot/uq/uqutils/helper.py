#!/usr/bin/env python
"""
Documentation for this imm module

More details.
"""


from copy import deepcopy
import numpy as np
import uuid
import os
import datetime
import pickle
from git import Repo
import datetime
import os
import json
import pickle
import dill

# for submodule in repo.submodules:
#     print(submodule)
#     diff=submodule.module().git.diff('HEAD~1..HEAD', name_only=True)
#     print(diff)




class DebugStatus:
    def __init__(self):
        self.a = 1

    def printstatus(self,debugStatuslist):
        leftlen = np.max([len(ss[0]) for ss in debugStatuslist])+5
        for i in range(len(debugStatuslist)):
            print(str(debugStatuslist[i][0]).ljust(leftlen,' '),' : ',str(debugStatuslist[i][1]))

    def writestatus(self,debugStatuslist,filepath):
        leftlen = np.max([len(ss[0]) for ss in debugStatuslist])+5
        with open(filepath,'w') as ff:
            for i in range(len(debugStatuslist)):
                ff.write( str(debugStatuslist[i][0]).ljust(leftlen,' ')+' : '+str(debugStatuslist[i][1])+'\n')

    def returnstatus(self,debugStatuslist):
        leftlen = np.max([len(ss[0]) for ss in debugStatuslist])+5

        ss=''
        for i in range(len(debugStatuslist)):
            ss=ss + str(debugStatuslist[i][0]).ljust(leftlen,' ') + ' : ' + str(debugStatuslist[i][1])+'\n'

        return ss

def deepcopy_with_sharing(obj, shared_attribute_names, memo=None):
    '''
    Deepcopy an object, except for a given list of attributes, which should
    be shared between the original object and its copy.

    obj is some object
    shared_attribute_names: A list of strings identifying the attributes that
        should be shared between the original and its copy.
    memo is the dictionary passed into __deepcopy__.  Ignore this argument if
        not calling from within __deepcopy__.
    '''
    assert isinstance(shared_attribute_names, (list, tuple))
    shared_attributes = {k: getattr(obj, k) for k in shared_attribute_names}

    if hasattr(obj, '__deepcopy__'):
        # Do hack to prevent infinite recursion in call to deepcopy
        deepcopy_method = obj.__deepcopy__
        obj.__deepcopy__ = None

    for attr in shared_attribute_names:
        del obj.__dict__[attr]

    clone = deepcopy(obj)

    for attr, val in shared_attributes.iteritems():
        setattr(obj, attr, val)
        setattr(clone, attr, val)

    if hasattr(obj, '__deepcopy__'):
        # Undo hack
        obj.__deepcopy__ = deepcopy_method
        del clone.__deepcopy__

    return clone



class A(object):

    def __init__(self):
        self.copy_me = []
        self.share_me = []

    def __deepcopy__(self, memo):
        return deepcopy_with_sharing(self, shared_attribute_names = ['share_me'], memo=memo)
