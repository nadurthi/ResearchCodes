# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as nplinalg
import numba as nb
import multiprocessing as mp
import queue
import utils.math.gen_seq as utmthgenseq
import uq.information.distance as uqinfodist
import time
import copy 
import random
import pdb


def exhaustive_seq_time(tvec,robots,targetset,Targetfilterer,searchMIwt=0.5):
    # remeber: tvec[0] is done and has optimized control
    # Minimize the control cost and maximize the information
    # propagate the prior targets to all time steps
    # robots[ri].statehistory[tvec[0]]
    
    for i in range(targetset.ntargs):
        targetset[i].freeze(recorderobj=False)
    for ri in  range(len(robots)):
        robots[ri].freezeState()    
    
    # pdb.set_trace()
    
    Ukfilter = [None]*targetset.ntargs
  
    
    # save the initial control history states of the robots
    for ri in range(len(robots)):
        for j in range(len(tvec)):
            robots[ri].statehistory.pop(tvec[j],None)
            robots[ri].controllerhistory.pop(tvec[j],None)
        robots[ri].statehistory[tvec[0]] = robots[ri].xk.copy()
    

    for ti in range(0,len(tvec)-1):
        dt = tvec[ti+1]-tvec[ti]
        # prop from tvec[ti] to tvec[ti+1]
        # update robots at tvec[ti+1]
        # find best combos of robot positions in tvec[ti+1]
        # then update the robot, do the measurement update at tvec[ti+1] with optimal 
        for i in range(targetset.ntargs):
            if targetset[i].isSearchTarget():
                targetset[i].filterer.propagate( tvec[ti],dt,targetset[i],updttarget=True)
            else:
                if targetset[i].isInActive(): 
                    continue
                uk = None
                Targetfilterer.propagate(tvec[ti],dt,targetset[i],uk,updttarget=True)
        
        for i in range(targetset.ntargs):
            targetset[i].freeze(recorderobj=False)
        
        Xsets=[]
        X0=[]
        for ri in range(len(robots)):
            x0 = robots[ri].xk
            
            Xset = robots[ri].reachableNodesFrom(x0,1.1*dt,returnNodeIds=True)
            ind0 = robots[ri].mapobj.getNodeIdx(x0[0:2])   
            X0.append(ind0)
            Xset.remove(ind0)
            Xsets.append(Xset)
            
        
        # print(X0)
        # print(Xsets)
        # pdb.set_trace()
           
        cnt=0
        m=-1e5
        bestseq=None
        for Xseq in utmthgenseq.independent_seq_generator(Xsets):
            # reset targets to prior at time tvec[ti+1]
            for i in range(targetset.ntargs):
                targetset[i].defrost(recorderobj=False)
            
            # update the robot positions to tvec[ti+1] and measure update
            for ri in range(len(robots)):
                idx = Xseq[ri]
                xk = robots[ri].mapobj.getNodefromIdx(idx)
                robots[ri].xk[0:2] = np.copy(xk)
                robots[ri].updateTime(tvec[ti+1])
                robots[ri].updateSensorModel()  

                # now do the measurement update
                sensormodel = robots[ri].sensormodel
                for i in range(targetset.ntargs):
                    if targetset[i].isSearchTarget():
                        targetset[i].filterer.measUpdate(tvec[ti+1], dt, targetset[i], sensormodel,None, updttarget=True)
                    else:
                        if targetset[i].isInActive(): 
                            continue
                        # compute the intersection of FOV and COV ellipsoid
                        covOverLap = sensormodel.intersectCovWithFOV(targetset[i].xfk,targetset[i].Pfk)
                        if covOverLap>0.3:
                            Targetfilterer.measUpdate(tvec[ti+1], dt, targetset[i], sensormodel, None, updttarget=True, fovSigPtFrac=-1)             
                        else:
                            Targetfilterer.measUpdate(tvec[ti+1], dt, targetset[i], sensormodel, None, updttarget=True, fovSigPtFrac=-1,justcopyprior=True)             
            
            # compute MI and MIsearch for this combo of sensor positions
            MIsearch = 0
            MI = 0
            for i in range(targetset.ntargs):
                if targetset[i].isSearchTarget():
                    pf=targetset[i].recorderprior.getvar_bytime('xfk',tvec[ti+1])
                    pu=targetset[i].recorderpost.getvar_bytime('xfk',tvec[ti+1])
                    pf[pf==0]=1e-3
                    pu[pu==0]=1e-3
                    Hf = -(pf*np.log(pf)+(1-pf)*np.log(1-pf))
                    Hu = -(pu*np.log(pu)+(1-pu)*np.log(1-pu))
                    dd = np.sum(Hf-Hu)
                    MIsearch = MIsearch+ dd
                    if np.iscomplex(dd):
                        print("---- complex search target probs----")
                        print(dd)
                        print(pf)
                        print(pu)
                else:
                    if targetset[i].isInActive(): 
                        continue
                    
                    Pfk=targetset[i].recorderprior.getvar_bytime('Pfk',tvec[ti+1])
                    Puk=targetset[i].recorderpost.getvar_bytime('Pfk',tvec[ti+1])
                    # pdb.set_trace()
                    mi = uqinfodist.mutualInformation_covs(Pfk,Puk)
                    MI=MI+mi
            
            # print(Xseq,MI,MIsearch)
            
            MIfull  = MI + searchMIwt*MIsearch
            if MIfull>m:
                bestseq = Xseq
                m = MIfull
                
            cnt+=1
        # print("X0 --> bestseq => ",X0,bestseq)
        
        # do the correct measurement update with optimal seq
        for i in range(targetset.ntargs):
            targetset[i].defrost(recorderobj=False)
        
        
        # Now set the robot controll history
        # uk_key=(idx0,idth0,idxf,idthf)
        for ri in range(len(robots)):
            x0 = robots[ri].statehistory[tvec[ti]].copy()
            ind0 = robots[ri].mapobj.getNodeIdx(x0[0:2])    
            # for uk_key in robots[ri].iterateControlsKeys():
            uk_key=(ind0,0,bestseq[ri],0)
            # print(tvec[ti],tvec[ti+1],ri,uk_key)
            s1 = robots[ri].mapobj.getNodefromIdx(uk_key[2])
            th1 = robots[ri].mapobj.getthfromIdx(uk_key[3])
            xk1 = np.hstack([s1,th1])
            robots[ri].controllerhistory[tvec[ti]]=uk_key
            robots[ri].statehistory[tvec[ti+1]]=xk1.copy()
            
            
            robots[ri].xk = xk1.copy()
            robots[ri].updateTime(tvec[ti+1])
            robots[ri].updateSensorModel() 
            
            # now do the measurement update with optimal robot positions
            sensormodel = robots[ri].sensormodel
            for i in range(targetset.ntargs):
                if targetset[i].isSearchTarget():
                    targetset[i].filterer.measUpdate(tvec[ti+1], dt, targetset[i], sensormodel,None, updttarget=True)
                else:
                    if targetset[i].isInActive(): 
                        continue
                    # compute the intersection of FOV and COV ellipsoid
                    covOverLap = sensormodel.intersectCovWithFOV(targetset[i].xfk,targetset[i].Pfk)
                    if covOverLap>0.3:
                        Targetfilterer.measUpdate(tvec[ti+1], dt, targetset[i], sensormodel, None, updttarget=True, fovSigPtFrac=-1)             
                    else:
                        Targetfilterer.measUpdate(tvec[ti+1], dt, targetset[i], sensormodel, None, updttarget=True, fovSigPtFrac=-1,justcopyprior=True)
                        
    # set the targets back to normal at tvec[0]
    for i in range(targetset.ntargs):
        for j in range(1,len(tvec)):
            targetset[i].recorderprior.deleteRecord(tvec[j])
            targetset[i].recorderpost.deleteRecord(tvec[j])
        targetset[i].resetState2timePostRecord(tvec[0])
        
    # set the robots back to normal at tvec[0]
    for ri in range(len(robots)):
        robots[ri].xk = robots[ri].statehistory[tvec[0]].copy()
        robots[ri].updateSensorModel()
        robots[ri].updateTime(tvec[0])

    # pdb.set_trace()