#!/usr/bin/env python
"""
Documentation for this imm module

More details.
"""



import numpy as np
import logging
import copy
import os
import pickle as pkl
import math
import pandas as pd
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from uq.uqutils import recorder as uqrecorder
from uq.motfilter import mot
from utils.math import arraycounters as myutac

# %% usage
# tgt = Target(dynModels=[],xf0=xf0,Pf0=Pf0,recordfilterstate=True,status='active')
# PDA.targetset.addTarget(tgt)


#%%
#scriptpath=os.path.dirname(os.path.abspath(__file__))

scriptpath = ''

def getNumEventsJPDA(Nt,Nm):
    if Nm>Nt:
        s=1
        for i in range(Nt):
            s=s+math.factorial(Nt)*math.factorial(Nm)/(math.factorial(i)*math.factorial(Nt-i)*math.factorial(Nm-Nt+i) )
        print(s)
        return s
    else:
        print("Nm>Nt")
        return None


def iterateJPDAevents(Nt,Nm,cache=True):
    """
    yield [(n1,m1),(n2,m2),(None,m3),(None,m4)]
    length is same as number of measurements
    """
    foldername = os.path.join(scriptpath,'JPDAevents')
    filename = os.path.join(scriptpath,'JPDAevents','Nt%d-Nm%d.pkl'%(Nt,Nm))
    if cache is True:
        if os.path.isfile(filename):
#            print("using cache file")
            with open(filename,'rb') as F:
                A = pkl.load(F)
            for i in range(len(A)):
                yield A[i]

            return


    A=[]
    dd = Nt*Nm
    for i in range(2**(dd)):
        a = list( format(i, '#0%db'%(dd+2))[2:] )
        a = np.array([int(b) for b in a])
        a = a.reshape(Nt,Nm)
        if any(np.sum(a,axis=1)>1) or any(np.sum(a,axis=0)>1):
            continue
        c=[]
        for j in range(Nm):
            k=np.argwhere(a[:,j]==1)
            if len(k)==0:
                c.append((None,j))
            else:
                c.append((int(k),j))
        A.append(c)
        yield c


    if cache is True:
        if not os.path.isdir(foldername):
            os.makedirs(foldername)
        with open(filename,'wb') as F:
            pkl.dump(A,F,protocol=pkl.HIGHEST_PROTOCOL)

def iterateJPDAevents_method2(Nt,Nm,cache=True):
    """
    yield [(n1,m1),(n2,m2),(None,m3),(None,m4)]
    length is same as number of measurements
    """
    foldername = os.path.join(scriptpath,'JPDAevents')
    filename = os.path.join(scriptpath,'JPDAevents','Nt%d-Nm%d.pkl'%(Nt,Nm))
    if cache is True:
        if os.path.isfile(filename):
#            print("using cache file")
            with open(filename,'rb') as F:
                A = pkl.load(F)
            for i in range(len(A)):
                yield A[i]

            return


    A=[]
    bmc = myutac.BinMatrixCounter(Nt,Nm)

    while True:
        a,rst=bmc.getAndIncrement()
        if rst>0:
            break

        c=[]
        for j in range(Nm):
            k=np.argwhere(a[:,j]==1)
            if len(k)==0:
                c.append((None,j))
            else:
                c.append((int(k),j))
        A.append(c)
        yield c


    if cache is True:
        if not os.path.isdir(foldername):
            os.makedirs(foldername)
        with open(filename,'wb') as F:
            pkl.dump(A,F,protocol=pkl.HIGHEST_PROTOCOL)




#%%


class JPDAMOT(mot.MOTfilter):
    """
    No data association, All targets get all the measurements
    This is for testing puposes

    """
    motname = 'JPDA'
    def __init__(self, filterer,recordMeasurementSets=True,recorderobjmeas=None,recordDA=True,recorderobjDA=None,PD = 0.8,V=1000,uf=None , Gamma=5 ):
        self.grounttruthDA = None
        super().__init__( filterer,recordMeasurementSets=recordMeasurementSets,recorderobjmeas=recorderobjmeas)
        if uf is None:
            self.uf=lambda phi: 1
        else:
            self.uf = uf

        self.V = V
        self.PD = PD
        self.Gamma =Gamma

        self.recordergrounttruthDA = uqrecorder.StatesRecorder_list(statetypes = ['DA'] )
        self.recordDA = recordDA
        if recorderobjDA is None:
            self.recorderDA = uqrecorder.StatesRecorder_list(statetypes = ['DA'] )
        else:
            self.recorderDA = recorderobjDA


    def setgrounttruthDA(self,t,dt, grounttruthDA,makecopy=True ):
        if makecopy:
            self.grounttruthDA = copy.deepcopy(grounttruthDA)
        else:
            self.grounttruthDA = grounttruthDA

        if self.recordDA:
            self.recordergrounttruthDA.record(t=t,DA=grounttruthDA)

    def set_DAmat_from_groundtruthDA(self,t,dt):
        self.associationDA = self.grounttruthDA

        if self.recordDA:
            self.recorderDA.record(t=t,DA=self.associationDA)




    def compute_DAmat(self,t,dt,Zkset):
        """
         Zkset should be {'sensID1':[zk1,zk2], 'sensID2':[zk1,zk2,zk3],}
         DAmat is a pandas dataframe with target IDs as index and columns and measurment index
        """
        Nt = self.targetset.ntargs



#        targetIDlist = self.targetset.targetIDs()
        DA= {} # betas
        PDFzs = {}
        Gatting={}
        mattargMeasLikelihood = {}

        for sensID in Zkset:
            Nm = len(Zkset[sensID])



            if sensID not in PDFzs:
                PDFzs[sensID] = []
                Gatting[sensID] = []

            for i in range(Nt):
                PDFzs[sensID].append( self.filterer.getPDFz(t,dt,self.targetset[i], self.sensorset.getbyID(sensID) ) )

            if sensID not in DA:
                DA[sensID] = np.zeros((Nt,Nm+1)) #pd.DataFrame(index =targetIDlist,columns= ['None'] + list(range( len(Zkset[sensID]) )) )
#                DA[sensID].fillna(0,inplace=True)

            if sensID not in mattargMeasLikelihood:
                mattargMeasLikelihood[sensID] = np.zeros((Nt,Nm)) #pd.DataFrame(index =targetIDlist,columns=range( len(Zkset[sensID])) )
#                Gating[sensID] =pd.DataFrame(index =targetIDlist,columns=range( len(Zkset[sensID])) )

            for m in range(len(Zkset[sensID])):
                zk = Zkset[sensID][m]
                for i in range(Nt):
                    mattargMeasLikelihood[sensID][i,m] = PDFzs[sensID][i].pdf(zk)
                    if PDFzs[sensID][i].isInNsig(zk,self.Gamma):
                        Gatting[sensID].append((i,m))



            for Ae in iterateJPDAevents_method2(Nt,Nm):
                pZAe = 1
                flgAeUseless=False


                for e in filter(lambda x: x[0] is not None, Ae):
                    if e not in Gatting[sensID]:
                        flgAeUseless = True
                        break
                    else:
                        pZAe = pZAe * mattargMeasLikelihood[sensID][e[0] ,e[1]]

                if flgAeUseless:
                    continue


#                sttime = time.time()
                # the number of clutter in the events
                phi = len([x for x in Ae if x[0] is None])
                Dt = Nm-phi
                pPhi =  math.factorial(phi)*self.uf(phi)/math.factorial(Nm)
                pAe = (1/self.V)**phi * self.PD**Dt * (1-self.PD)**(Nt-Dt) * pPhi

#                pZAe = 1
#                for e in filter(lambda x: x[0] is not None, Ae):
#                    pZAe = pZAe * dftargMeasLikelihood[sensID].loc[targetIDlist[ e[0] ],e[1]]

                pAeIk = pAe * pZAe
#                print("t2 = ", time.time()-sttime )

#                sttime = time.time()
                Aetargs = [e[0] for e in Ae if e[0] is not None]
                #update the null probabalities
                for i in range(Nt):
                    if i not in Aetargs:
                        DA[sensID][i,0] = DA[sensID][i,0]+pAeIk
#                print("t3 = ", time.time()-sttime )

#                sttime = time.time()
                for e in filter(lambda x: x[0] is not None,Ae):
                    DA[sensID][ e[0] ,e[1]+1] = DA[sensID][ e[0] ,e[1]+1] + pAeIk

#                print("t4 = ", time.time()-sttime )
#            DA[sensID]= np.round(DA[sensID], decimals=2)

#            DA[sensID] = pd.DataFrame(DA[sensID],index =targetIDlist,columns= ['None'] + list(range( len(Zkset[sensID]) )) )
            d=np.sum(DA[sensID],axis=1)
#            d=DA[sensID].sum(axis=1)
            DA[sensID]=np.divide(DA[sensID],d[:,np.newaxis])
#            print(DA[sensID].round(decimals=2).values)
        self.associationDA = DA
        if self.recordDA:
            self.recorderDA.record(t=t,DA=self.associationDA)



    def propagate(self,t, dt, Uk,  **kwargs):
        self.currt = t + dt


        if Uk is None:
            Uk = [None]*self.targetset.ntargs


        for i in range(self.targetset.ntargs):
            if self.targetset[i].isactive():
                # self.targetset[i].propagate(tk,dtk,Uk[i])
                self.filterer.propagate(t,dt,self.targetset[i],Uk[i],
                                        updttarget=True, **kwargs)


    def measUpdate(self,t,dt, Zkset, *args, **kwargs):
        """
         Zkset should be {'sensID1':[zk1,zk2], 'sensID2':[zk1,zk2,zk3],}
         for now only 1 sensor works
        """
        if len(Zkset)>1:
            raise NotImplementedError("len(Zkset)>1 not implemented")

        self.filterstage = 'Measurement Updated'


        # sensorIDs = self.sensorset.sensorIDs()

        if self.sensorset.nsensors>1:
            raise NotImplementedError('Only 1 sensor assimilation is implemented')
        else:
            sensidx = 0
            sensID = self.sensorset[sensidx].ID

        for i in range(self.targetset.ntargs):
            if self.targetset[i].isactive():

#                targID = self.targetset[i].ID
#                betas = self.associationDA[sensID].loc[targID,:].values
                betas = self.associationDA[sensID][i,:]
                # append null/false positive beta at beginging
#                betas = np.hstack([0,betas])
                Zk = Zkset[sensID]
                self.filterer.measWeightedBetasUpdt(t,dt,self.targetset[i],
                                          self.sensorset[sensidx],
                                          betas,Zk, updttarget=True, **kwargs)

    def debugStatus(self):
        ss = []
        ss.append( ['----------','----------'] )
        ss.append( [self.motname+' ID',self.ID ] )
        ss.append( ['----------','----------'] )
        ss.append( ['Current time', self.currt] )
        ss.append( ['Record Mesurementsets', self.recordMeasurementSets] )
        ss.append( ['filterer', self.filterer.filterName] )
#        ss.append( ['Dynamical Model idx',self.dynmodelIdx] )
#        ss.append( ['GMM comp idx',self.gmmIdx] )
#        ss.append( ['Sensor model idx',self.sensorModelIdx] )

        ss = ss + [['\t'+s1[0],s1[1]] for s1 in self.targetset.debugStatus()]
        ss = ss + [['\t'+s1[0],s1[1]] for s1 in self.sensorset.debugStatus()]


        return ss

        # add tab






#%%
