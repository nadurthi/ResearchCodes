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

def traj_filter(XSET,methods=['ExactSetNodes','>=90degTurn']):
    for mm in methods:
        if mm == 'ExactSetNodes':
            Xset=[]
            X=[]
            for Xseq in XSET:
                ss = set(Xseq)
                if ss not in X:
                    Xset.append(Xseq)
                    X.append(ss)
            
            XSET = Xset
    
    return XSET
        
def random_seq_robot(tvec,robots,targetset,Targetfilterer,searchMIwt=0.5):
    """
    sequential robot path generate
    for debug purposes just give out a random seq
    !!!!!!!!! THE ROBOTS ARE AT tvec[0] !!!!!!!!!!!!!!!!!
    """
    for i in range(targetset.ntargs):
        targetset[i].freeze(recorderobj=False)
    for r in  range(len(robots)):
        robots[r].freezeState()  
        
    # Ukfilter = [None]*targetset.ntargs
    # # propagate targets by one step
    # for j in range(len(tvec)-1):
    #     for i in range(targetset.ntargs):
    #         Targetfilterer.propagate(tvec[j],tvec[j+1]-tvec[j],
    #                                       targetset[i],
    #                                       Ukfilter[i],
    #                                       updttarget=True)
    
    # save the initial states of the robots
    for r in range(len(robots)):
        for j in range(len(tvec)):
            robots[r].statehistory.pop(tvec[j],None)
            robots[r].controllerhistory.pop(tvec[j],None)
        robots[r].statehistory[tvec[0]] = robots[r].xk.copy()
    
    optimalSeqs={}
    seqL = len(tvec)
    for r in range(len(robots)):
        dt = tvec[1]-tvec[0]
        T= tvec[-1]-tvec[0]
        x0 = robots[r].xk
        Xset = robots[r].reachableNodesFrom(x0,T+0.1*dt,returnNodeIds=True)
        
        ind0 = robots[r].mapobj.getNodeIdx(x0[0:2])    
        # now build constraint set for each node in Xset
        Xconst={}
        
        for idx in Xset:
            x = robots[r].mapobj.getNodefromIdx(idx)    
            Xconst[idx] = robots[r].reachableNodesFrom(x,1.1*dt,returnNodeIds=True)
            Xconst[idx].remove(idx)
        # pdb.set_trace()
        cnt=0
        XX=[]
        for Xseq in utmthgenseq.dependent_seq_generator(seqL,ind0,Xset,Xconst):
            XX.append(Xseq)
            cnt+=1
            
        ii = random.randint(0,len(XX)-1)
        optimalSeqs[r] = XX[ii]
    
    # Xseq has the node sequence ids     
        
    # Now set the robot controll history
    # uk_key=(idx0,idth0,idxf,idthf)
    for k in range(len(tvec)-1):
        for r in range(len(robots)):
            x0 = robots[r].statehistory[tvec[k]]
            # for uk_key in robots[r].iterateControlsKeys():
            uk_key=(optimalSeqs[r][k],0,optimalSeqs[r][k+1],0)
                
            s1 = robots[r].mapobj.getNodefromIdx(uk_key[2])
            th1 = robots[r].mapobj.getthfromIdx(uk_key[3])
            xk1 = np.hstack([s1,th1])
            robots[r].controllerhistory[tvec[k]]=uk_key
            robots[r].statehistory[tvec[k+1]]=xk1
    
        
    # set the targets back to normal at tvec[0]
    for i in range(targetset.ntargs):
        targetset[i].defrost(recorderobj=False)
        for j in range(1,len(tvec)):
            targetset[i].recorderprior.deleteRecord(tvec[j])
            
    # set the robots back to normal at tvec[0]
    for r in range(len(robots)):
        robots[r].xk = robots[r].statehistory[tvec[0]].copy()
        robots[r].updateSensorModel()
        
def exhaustive_seq_robot(tvec,robots,targetset,Targetfilterer,searchMIwt=0.5):
    # remeber: tvec[0] is done and has optimized control
    # Minimize the control cost and maximize the information
    # propagate the prior targets to all time steps
    # robots[r].statehistory[tvec[0]]
    
    for i in range(targetset.ntargs):
        targetset[i].freeze(recorderobj=False)
    for r in  range(len(robots)):
        robots[r].freezeState()    
    
    # ctx = mp.get_context('spawn')
    ctx = mp
    seqQ = ctx.Queue()
    resQ = ctx.Queue()
    ExitFlag = ctx.Event()
    ExitFlag.clear()


    Ncore = 8
    processes = []
    for i in range(Ncore):
        p = ctx.Process(target=robotTargetInfo_seqrobot, args=(ExitFlag,seqQ,resQ,robots,targetset,tvec,Targetfilterer))
        processes.append( p )
        p.start()
        # print("created thread")    
        time.sleep(0.02)    
    
        
    # targetset[i].freeze_time
    
    
        

    Ukfilter = [None]*targetset.ntargs
  
    
    # save the initial control history states of the robots
    for r in range(len(robots)):
        for j in range(len(tvec)):
            robots[r].statehistory.pop(tvec[j],None)
            robots[r].controllerhistory.pop(tvec[j],None)
        robots[r].statehistory[tvec[0]] = robots[r].xk.copy()
    
    optimalSeqs={}
    for r in range(len(robots)):


        for ri in range(len(robots)):
            robots[ri].defrostState() 
            robots[ri].updateSensorModel()      
            


        seqL = len(tvec)
        dt = tvec[1]-tvec[0]
        T= tvec[-1]-tvec[0]
        x0 = robots[r].xk
        Xset = robots[r].reachableNodesFrom(x0,T+0.1*dt,returnNodeIds=True)
        ind0 = robots[r].mapobj.getNodeIdx(x0[0:2])    
        # now build constraint set for each node in Xset
        Xconst={}
        
        for idx in Xset:
            x = robots[r].mapobj.getNodefromIdx(idx)    
            Xconst[idx] = robots[r].reachableNodesFrom(x,1.1*dt,returnNodeIds=True)
            Xconst[idx].remove(idx)
        
        cnt=0
        XSET=[]
        for Xseq in utmthgenseq.dependent_seq_generator(seqL,ind0,Xset,Xconst):
            XSET.append(Xseq)
        
        XSET = traj_filter(XSET)
        
        for Xseq in XSET:
            XXseq = copy.deepcopy(optimalSeqs)
            XXseq[r]=Xseq
            seqQ.put(XXseq)
            
            
            # ExitFlag.set()
            # robotTargetInfo_seqrobot(ExitFlag,seqQ,resQ,robots,targetset,tvec,Targetfilterer)
            
            
            cnt+=1
            
        m=-1e5
        bestseq=None
        rescnt=0
        while True:
            res=None
            try:
                res=resQ.get(True,1)
            except queue.Empty:
                time.sleep(1)
            
            if res is not None:
                rescnt+=1
                XXseq = res[0]
                MI = res[1]
                MIsearch = res[2]
                cc = MI + MIsearch
                MIfull  = MI+searchMIwt*MIsearch
                if MIfull>m:
                    bestseq = XXseq
                    m = MIfull
                
            if rescnt==cnt:
                break
            print("Still receiving = (%d,%d) and (%d,%d)"%(rescnt,cnt,MI,MIsearch))
        optimalSeqs =copy.deepcopy( bestseq )
        
    ExitFlag.set()
    
        
        
    # Now set the robot controll history
    # uk_key=(idx0,idth0,idxf,idthf)
    for k in range(len(tvec)-1):
        for r in range(len(robots)):
            x0 = robots[r].statehistory[tvec[k]]
            # for uk_key in robots[r].iterateControlsKeys():
            uk_key=(optimalSeqs[r][k],0,optimalSeqs[r][k+1],0)
                
            s1 = robots[r].mapobj.getNodefromIdx(uk_key[2])
            th1 = robots[r].mapobj.getthfromIdx(uk_key[3])
            xk1 = np.hstack([s1,th1])
            robots[r].controllerhistory[tvec[k]]=uk_key
            robots[r].statehistory[tvec[k+1]]=xk1.copy()
    
        
    # set the targets back to normal at tvec[0]
    for i in range(targetset.ntargs):
        targetset[i].defrost(recorderobj=False)
        for j in range(1,len(tvec)):
            targetset[i].recorderprior.deleteRecord(tvec[j])
            targetset[i].recorderpost.deleteRecord(tvec[j])
        targetset[i].resetState2timePostRecord(tvec[0])
        
    # set the robots back to normal at tvec[0]
    for r in range(len(robots)):
        robots[r].xk = robots[r].statehistory[tvec[0]].copy()
        robots[r].updateSensorModel()
    
    # pdb.set_trace()
    
    for i in range(len(processes)):
        processes[i].join()


def robotTargetInfo_seqrobot(ExitFlag,seqQ,resQ,robots,targetset,tvec,Targetfilterer):
    """
    XXseq={0:Xseq,1:0:Xseq,...}
    if key in XXseq, then there is a sequence for this robot
    """
    for i in range(targetset.ntargs):
        targetset[i].freeze(recorderobj=False)
    for r in  range(len(robots)):
        robots[r].freezeState()    
    
    # print("starting thread")
    
    dt = tvec[1]-tvec[0]
    
    while(True):
        XXseq = None
        try:
            XXseq = seqQ.get(True,0.2)
        except queue.Empty:
            XXseq = None
        
        
        if XXseq is not None:
            for i in range(targetset.ntargs):
                targetset[i].defrost(recorderobj=False)
            for ri in range(len(robots)):
                robots[ri].defrostState() 
                robots[ri].updateSensorModel()   
                
            # CHECK!!!!!!!! targetset is at tvec[0]
            for ti in range(1,len(tvec)):
                # propagate
                t = tvec[ti]
                
                for i in range(targetset.ntargs):
                    if targetset[i].isSearchTarget():
                        targetset[i].filterer.propagate( t-dt,dt,targetset[i],updttarget=True)
                    else:
                        if targetset[i].isInActive(): 
                            continue
                        uk = None
                        Targetfilterer.propagate(t-dt,dt,targetset[i],uk,updttarget=True)
                    
                for r in  range(len(robots)):
                    if r in XXseq:
                        Xseq = XXseq[r]
                        idx = Xseq[ti]
                        xk = robots[r].mapobj.getNodefromIdx(idx)
                        robots[r].xk[0:2] = xk 
                        robots[r].updateSensorModel()
                        sensormodel = robots[r].sensormodel
                        for i in range(targetset.ntargs):
                            
                            if targetset[i].isSearchTarget():
                                targetset[i].filterer.measUpdate(t, dt, targetset[i], sensormodel,None, updttarget=True)
                            else:
                                if targetset[i].isInActive(): 
                                    continue
                                # compute the intersection of FOV and COV ellipsoid
                                covOverLap = sensormodel.intersectCovWithFOV(targetset[i].xfk,targetset[i].Pfk)
                                if covOverLap>0.3:
                                    Targetfilterer.measUpdate(t, dt, targetset[i], sensormodel, None, updttarget=True, fovSigPtFrac=-1)             
                                else:
                                    Targetfilterer.measUpdate(t, dt, targetset[i], sensormodel, None, updttarget=True, fovSigPtFrac=-1,justcopyprior=True)             
            MI=0        
            MIsearch=0
            for ti in range(1,len(tvec)):
                t = tvec[ti]
                for i in range(targetset.ntargs):
                    if targetset[i].isSearchTarget():
                        pf=targetset[i].recorderprior.getvar_bytime('xfk',t)
                        pu=targetset[i].recorderpost.getvar_bytime('xfk',t)
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
                        
                        Pfk=targetset[i].recorderprior.getvar_bytime('Pfk',t)
                        Puk=targetset[i].recorderpost.getvar_bytime('Pfk',t)
                        # pdb.set_trace()
                        mi = uqinfodist.mutualInformation_covs(Pfk,Puk)
                        MI=MI+mi
                        if np.iscomplex(mi):
                            print("---- complex target probs----")
                            print(nplinalg.eig(Pfk))
                            print(nplinalg.eig(Puk))
                            
            resQ.put((XXseq,MI,MIsearch))
            
        if (ExitFlag.is_set() and seqQ.empty()):
            break
    
    # print("Exiting thread")

        

                    