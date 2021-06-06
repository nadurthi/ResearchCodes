#!/usr/bin/env python
"""
Documentation for this imm module

More details.
"""



import numpy as np
import logging
import uuid
import copy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)




class Recorder:
    """
    time t is always a variable
    """
    recorder_datatype = 'base recorder'
    def __init__(self,**kwargs):
        self.ID = uuid.uuid4()
        self.Nextd=100
        self.data = {}
        self.idx = 0
        self.data['t'] = np.zeros(self.Nextd)
        self.endid = 0

        self.currN = self.Nextd



    @property
    def t(self):
        return self.data['t'][:self.idx].copy()

    def record(self,**kwargs):
        raise NotImplementedError("Method not implemented")

    def getitem_bytime(self,t):

        k = np.where(self.data['t'][:self.idx]==t)
        if len(k[0])==0:
            print("Index error: time t is not there")
            return None
        return self.__getitem__(k)


    def getitem_alltimes(self):
        ss={}
        for var in self.data:
            ss[var] = copy.deepcopy( self.data[var][:self.idx] )

        return ss

    def getvar_alltimes(self,var):
        dd = copy.deepcopy( self.data[var][:self.idx] )
        if isinstance(dd,(list,tuple)):
            try:
                dd = np.stack(dd,axis=0)
            except:
                pass

        return dd

    def getvar_byidx(self,var,idx):
        return copy.deepcopy( self.data[var][idx] )

    def getvar_bytime(self,var,t):
        k = np.where(self.data['t'][:self.idx]==t)
        if len(k[0])==0:
            return None
        return copy.deepcopy( self.data[var][k] )

    def __getitem__(self,k):
        if k>=self.idx:
            print("Index error in recorder k>=idx")
        ss=[]
        for var in self.data:
            ss.append( self.data[var][k].copy() )

        return tuple(ss)

    def makecopy(self):
        return copy.deepcopy(self)

    def getitem_bytimestep(self,k):
        raise NotImplementedError("Method not implemented")

    def cleardata(self,keepInitial=True):
        raise NotImplementedError("Method not implemented")

    def save(filepath,metadata):
        """
        save as pickle
        """
        raise NotImplementedError("Method not implemented")
        
    def debugStatus(self):
        ss=[]
        ss.append( ['----------','----------'] )
        ss.append( ['Recorder',self.ID] )
        ss.append( ['----------','----------'] )
        ss.append( ['recorder_datatype',self.recorder_datatype] )
        ss.append( ['statetypes',self.statetypes] )
        ss.append( ['idx',self.idx] )
        ss.append( ['currN',self.currN] )
        ss.append( ['endid',self.endid] )

        return ss

class StatesRecorder_fixedDim(Recorder):
    recorder_datatype = 'fixed numpy dims'
    def __init__(self,statetypes,Nextd=100):
        """
        statetypes = {'xfk':(5,),'Pfk':(5,5)}

        """
        super().__init__()

        self.statetypes = statetypes
        self.states = list(self.statetypes.keys())


        for var in self.statetypes:
            self.data[var] = np.zeros((self.Nextd,*self.statetypes[var] ))



    def debugStatus(self):
        ss = super().debugStatus()

        for var in self.data:
            ss.append( [var+'.shape',self.data[var].shape] )
        return ss

    def cleardata(self,keepInitial = True):
        if keepInitial is True:
            p = 1
        else:
            p = 0

        self.data['t'][p:]=self.data['t'][p:]*0

        for var in self.statetypes:
            self.data[var][p:] = self.data[var][p:]*0


        self.idx = p
        self.currN = self.Nextd
        self.endid = p



    def record(self, t,**kwargs):

        if self.idx == len(self.data['t'] ):
            self.data['t'] = np.hstack((self.data['t'],np.zeros(self.Nextd)))
            for var in self.statetypes:
                self.data[var] = np.concatenate(
                    (self.data[var], np.zeros((self.Nextd,*self.statetypes[var] ))), axis=0)

            self.currN += self.Nextd

        self.data['t'][self.idx] = t

        for var in kwargs:
            if var not in self.data:
                raise KeyError("Input var has to be initialized  for recorder")
            self.data[var][self.idx] = kwargs[var].copy()

        self.idx += 1
        self.endid = self.idx

class StatesRecorder_list(Recorder):
    recorder_datatype = 'list'
    def __init__(self,statetypes):
        """
        statetypes = ['xfk','Pfk']

        """
        self.ID = uuid.uuid4()
        self.data = {}

        self.data['t'] = []
        

        if not isinstance(statetypes,(list,tuple)):
            raise TypeError('statetypes have to be list or tuple')

        self.statetypes = statetypes
        self.states = self.statetypes

        for var in self.statetypes:
            self.data[var] = []

    def debugStatus(self,printit=False):
        ss=[]
        ss.append( ['----------','----------'] )
        ss.append( ['Recorder',self.ID] )
        ss.append( ['----------','----------'] )
        ss.append( ['recorder_datatype',self.recorder_datatype] )
        ss.append( ['statetypes',self.statetypes] )
        
        for var in self.data:
            ss.append( [var+'.len',len(self.data[var])] )
        
        if printit:
            print("\n".join(ss))
            
        return ss

    def cleardata(self,keepInitial = True):
        if keepInitial is True:
            p = 1
        else:
            p = 0

        self.data['t'][p:]=[]

        for var in self.statetypes:
            self.data[var][p:] = []

    
    def deleteRecord(self,t):
        try:
            k = self.data['t'].index(t)
        except ValueError:
            return False
        
        for var in self.statetypes:
            self.data[var].pop(k)
        self.data['t'].pop(k)
        
        return True
    
            
    def getvar_bytime(self,var,t):
        try:
            k = self.data['t'].index(t)
        except ValueError:
            return None

        return copy.deepcopy( self.data[var][k] )
    
    def getvar_list(self,var,t0,tf,returntimes=False):
        k=[i for i in range(len(self.data['t'])) if self.data['t'][i]>=t0 and self.data['t'][i]<=tf]
        st=[]
        ss=[]
        for i in k:
            st.append(self.data['t'][i]) 
            ss.append(self.data[var][i]) 
        
        if returntimes:
            return st,ss
        else:
            return ss
        
    def getvar_alltimes_stacked(self,var):

        return np.stack( self.data[var],axis=0 )

    def getvar_uptotime_list(self,var,t,returntimes=False):
        k=[i for i in range(len(self.data['t'])) if self.data['t'][i]<=t]
        if len(k)==0:
            if returntimes:
                return None,None
            else:
                return None
        else:
            k=k[-1]
        
        tt = self.data['t'][:k+1]
        ss = copy.deepcopy( self.data[var][:k+1] )
        
        if returntimes:
            return tt,ss
        else:
            return ss
    

        
        
    def getvar_uptotime_stacked(self,var,t,returntimes=False):
        
        k=[i for i in range(len(self.data['t'])) if self.data['t'][i]<=t]
        if len(k)==0:
            if returntimes:
                return None,None
            else:
                return None
        else:
            k=k[-1]
        
                    
        ss = np.stack( self.data[var][:k+1],axis=0 )
        st = np.hstack( self.data['t'][:k+1] )
        
        if returntimes:
            return st,ss
        else:
            return ss
    
#    def getgmm_stacked
    def record(self, t,updateIfExists=True, **kwargs):
        
        try:
            k = self.data['t'].index(t)
        except ValueError:
            k = None
        
        # print("recorder index k = ",k)
        if k is not None: # record exists
            if updateIfExists is False:
                raise Exception("record exists for time: ",t," : cannot update the record")
            # print("updating record as it exists")
            
            # update data
            for var in kwargs:
                if var in self.data:
                    self.data[var][k] = copy.deepcopy(kwargs[var]) 
    

                    
        else:
            # if not there, then append ---------
            # append time
               
            self.data['t'].append(t)
            
            # append data
            for var in kwargs:
                if var not in self.data:
                    self.data[var] = [None]*(len(self.data['t'])-1)
                self.data[var].append( copy.deepcopy(kwargs[var]) )
    
            for var in self.data:
                if var not in kwargs and var != 't':
                    self.data[var].append( None )
    
            

    def recordupdate(self, t, **kwargs):
        raise NotImplementedError("Deprecated ")
        
        # replace value if t exists just replace the value
        k = np.where(self.data['t'][:self.idx]==t)
        # print("recorder index k = ",k)
        if len(k[0])>0:
            # print("updating record as it exists")
            idx = k[0][0]
            # update data
            for var in kwargs:
                if var in self.data:
                    self.data[var][idx] = copy.deepcopy(kwargs[var]) 
    

                    
        else:
            # if not there, then append ---------
            # append time
            if self.idx == len(self.data['t'] ):
                self.data['t'] = np.hstack((self.data['t'],np.zeros(self.Nextd)))
                self.currN += self.Nextd
    
            self.data['t'][self.idx] = t
            
            # append data
            for var in kwargs:
                if var not in self.data:
                    self.data[var] = [None]*self.idx
                self.data[var].append( copy.deepcopy(kwargs[var]) )
    
            for var in self.data:
                if var not in kwargs and var != 't':
                    self.data[var].append( None )
    
            self.idx += 1





