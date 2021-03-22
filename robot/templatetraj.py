import numpy as np
import numpy.linalg as nplnalg
import scipy.linalg as sclnalg #block_diag
import scipy.optimize as scopt #block_diag
import utils.redis as utred
import pickle as pkl
import time

def propagator(x0,Uk,tvec,robotdyn):
    n = np.floor(Uk.shape[0]/2).astype(int)
    # print(n,Uk.shape[0])
    vmags = Uk[0:n]
    Omegas = Uk[n:]
    xk=x0
    X=[]
    X.append(xk)
    for i in range(len(tvec)-1):
        tt, xk = robotdyn.propforward(tvec[i], tvec[i+1]-tvec[i], xk, np.array([vmags[i],Omegas[i]])) 
        X.append(xk)
    X=np.array(X)
    return X


    
def funcOpt(Uk,x0,xf,tvec,robotdyn):
    X = propagator(x0,Uk,tvec,robotdyn)
    
    err = nplnalg.norm( xf-X[-1] ) +0.01*nplnalg.norm( nplnalg.norm(X[:,2:4],axis=1) )
    
    return err

def optimizeTraj(robID,x0,xf,U0): #,constraints

    n = np.floor(U0.shape[0]/2).astype(int)
    const1 = lambda Uk: np.hstack([Uk[:n]-NominalVmag,-Uk[:n]+MaxVmag,-Uk[n:]+MaxOmega,Uk[n:]+MaxOmega])
    constraints = {'type':'ineq','fun':const1}
    
    robotobj = pkl.loads(utred.rediscon.hget(robID,'robotobj'))
    robotdyn = robotobj.dynModel
    MaxVmag = robotobj.MaxVmag #5
    MaxOmega = robotobj.MaxOmega #3
    NominalVmag= robotobj.NominalVmag #2
    
    tvec = pkl.loads(utred.rediscon.hget(robID,'tvec'))
    
    
    res = scopt.minimize(funcOpt, U0,args=(x0,xf,tvec,robotdyn),constraints=constraints)
    return res

def generateTemplates(robotobj,dt,T):
    """
    

    Parameters
    ----------
    mapobj : TYPE
        DESCRIPTION.
    robotgrid : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.
    robotdyn : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    mapobj = robotobj.mapobj
    
    robotdyn = robotobj.dynModel
    MaxVmag = robotobj.MaxVmag #5
    MaxOmega = robotobj.MaxOmega #3
    NominalVmag= robotobj.NominalVmag #2
    
    tvec=np.arange(0,T,dt)
    
    robID = str(robotobj.ID)
    utred.rediscon.hset(robID,'robotobj',pkl.dumps(robotobj, protocol=pkl.HIGHEST_PROTOCOL))
    utred.rediscon.hset(robID,'tvec',pkl.dumps(tvec, protocol=pkl.HIGHEST_PROTOCOL))
    
    n=len(tvec)-1
    const1 = lambda Uk: np.hstack([Uk[:n]-NominalVmag,-Uk[:n]+MaxVmag,-Uk[n:]+MaxOmega,Uk[n:]+MaxOmega])
    constraints = {'type':'ineq','fun':const1}

    Djob={}
    for idx0,x0pos in mapobj.iteratenodes():
        for idth0,th0 in mapobj.iteratedirn(x0pos):
            for idxf,xfpos in mapobj.iteratenodes():
                for idthf,thf in mapobj.iteratedirn(xfpos):
                    if nplnalg.norm(x0pos-xfpos)>MaxVmag*T:
                        continue
                    uk_key=(idx0,idth0,idxf,idthf)
                    
                    v0 = NominalVmag*np.array([np.cos(th0),np.sin(th0)])
                    x0 = np.hstack([x0pos,v0,0])
                    vf = NominalVmag*np.array([np.cos(thf),np.sin(thf)])
                    xf = np.hstack([xfpos,vf,0])
                    U0 = np.hstack([MaxVmag/2*np.ones(n),np.zeros(n)])
                    # res = scopt.minimize(funcOpt, U0,args=(x0,xf,tvec,robotdyn),constraints={'type':'ineq','fun':const1})
                    Djob[uk_key] = {'thf':thf,'xf':xf,'x0':x0,'job':utred.redisQ.enqueue(optimizeTraj,args=(robID,x0,xf,U0), result_ttl=86400)} #,
                    
                    # Uopt = res.x
                    # cost = res.fun
                    # Xtraj = propagator(x0,Uopt,tvec,robotdyn)
                    # if nplnalg.norm(Xtraj[-1,0:2]-xfpos)>robotobj.MinTempXferr:
                    #     continue
                    
                    
                    # mapobj.addTemplateTraj(uk_key=uk_key,val={'xfpos':xfpos,'thf':thf,'Xtraj':Xtraj,'cost':cost} )
                    # print("done with: ",(idx0,idth0,idxf,idthf))
                    # print("success: ",res.success)
                    # print("success: ",res.message)
                    # mapobj.plotdebugTemplate(uk_key)
                    
    
    
    
    while True:
        isalldone = []
        for uk_key in Djob:
            if Djob[uk_key]['job'].result is None: 
                time.sleep(5)
                isalldone.append(False)
            else:
                isalldone.append(True)
        if np.all(isalldone):
            break
        
        
    for uk_key in Djob:
        if Djob[uk_key]['job'].result is None: 
            time.sleep(5)
        else:
            res = Djob[uk_key]['job'].result
            Uopt = res.x
            cost = res.fun
            Xtraj = propagator(Djob[uk_key]['x0'],Uopt,tvec,robotdyn)
            if nplnalg.norm(Xtraj[-1,0:2]-Djob[uk_key]['xf'][0:2])>robotobj.MinTempXferr:
                continue
            
            
            robotobj.addTemplateTraj(uk_key=uk_key,val={'xfpos':Djob[uk_key]['xf'][0:2],'thf':Djob[uk_key]['thf'],'Xtraj':Xtraj,'cost':cost} )
            robotobj.plotdebugTemplate(uk_key)
                    
