import numpy as np
import robot.gridrobot as robgrid
from uq.information import distance as uqinfodis
import collections as clc
import copy
from utils.math import geometry as utmthgeom
from uq.gmm import gmmbase as uqgmmbase
from sensortasking import robotinfo as sensrbinfo

    
def dynamicProgRobot2Dgrid_SeqRobot(simmanager,tvec,robots,targetset,Targetfilterer,infoconfig,splitterConfig,mergerConfig):
    # remeber: tvec[0] is done and has optimized control
    # Minimize the control cost and maximize the information
    # propagate the prior targets to all time steps
    simmanager.pickledata([tvec,robots,targetset,Targetfilterer,infoconfig,splitterConfig,mergerConfig],
                          ['debugdata','dynamicProgRobot2Dgrid_SeqRobot'],str(int(tvec[0]))+'.pkl')
    
    Ukfilter = [None]*targetset.ntargs
    InfoCosts=[]
    Cost2gos=[]
    
    # propagate targets by one step
    for j in range(len(tvec)-1):
        for i in range(targetset.ntargs):
            Targetfilterer.propagate(tvec[j],tvec[j+1]-tvec[j],
                                          targetset[i],
                                          Ukfilter[i],
                                          updttarget=True)
    
    # save the initial states of the robots
    for r in range(len(robots)):
        for j in range(len(tvec)):
            robots[r].statehistory.pop(tvec[j],None)
            robots[r].controllerhistory.pop(tvec[j],None)
        robots[r].statehistory[tvec[0]] = robots[r].xk.copy()
        
    # doing DP for robots in seq and from end time
    for r in range(len(robots)):
        ridstates = robots[r].mapobj.XYgvec
        infocost = sensrbinfo.robotTargetInfo(simmanager,r,ridstates,robots,targetset,tvec,Targetfilterer,infoconfig,splitterConfig,mergerConfig)
        
        J=clc.defaultdict(dict)
        for k in range(len(tvec)-1,-1,-1):
            if k == len(tvec)-1:
                for n in range(ridstates.shape[0]):
                    s = ridstates[n]
                    for idxth,th in robots[r].mapobj.iteratedirn(s):
                        xk = np.hstack([s,th])
                        J[tvec[k]][tuple(xk)] = {'cost':infocost[tvec[k]][n],'uk_key':None}
            else:
                for n in range(ridstates.shape[0]):
                    s = ridstates[n]
                    for idxth,th in robots[r].mapobj.iteratedirn(s):
                        xk = np.hstack([s,th])
                        mkopt = 1e5
                        ukopt = None
                        for uk_key in robots[r].iterateControlsKeys(s,th):
                            s1 = robots[r].mapobj.getNodefromIdx(uk_key[2])
                            th1 = robots[r].mapobj.getthfromIdx(uk_key[3])
                            xk1 = np.hstack([s1,th1])
                            ucost = robots[r].gettemplateCost(uk_key)
                            M = 0.001*ucost - 100*infocost[tvec[k]][n] + J[tvec[k+1]][tuple(xk1)]['cost']
                            if M<mkopt:
                                mkopt = M
                                ukopt = uk_key
                        J[tvec[k]][tuple(xk)] = {'cost':mkopt,'uk_key':ukopt}
        
        InfoCosts.append(infocost)
        Cost2gos.append(J)
        
        # Now set the robot controll history
        for k in range(len(tvec)-1):
            x0 = robots[r].statehistory[tvec[k]]
            uk_key= J[tvec[k]][tuple(x0)]['uk_key']
            s1 = robots[r].mapobj.getNodefromIdx(uk_key[2])
            th1 = robots[r].mapobj.getthfromIdx(uk_key[3])
            xk1 = np.hstack([s1,th1])
            robots[r].controllerhistory[tvec[k]]=uk_key
            robots[r].statehistory[tvec[k+1]]=xk1
    
        
    # set the targets back to normal at tvec[0]
    for j in range(1,len(tvec)):
        for i in range(targetset.ntargs):
            targetset[i].recorderprior.deleteRecord(tvec[j])
            
    # set the robots back to normal at tvec[0]
    for r in range(len(robots)):
        robots[r].xk = robots[r].statehistory[tvec[0]].copy()
        robots[r].updateSensorModel()
    
    return InfoCosts, Cost2gos