import numpy as np
import numpy.linalg as nplnalg
import scipy.linalg as sclnalg
from uq.quadratures import cubatures as uqcub
from uq.gmm import gmmbase as uqgmmbase
from uq.gmm import splitter as uqsplit

def splitGMM_ryanruss(gmm,f,N,ruleoption,maxlvl):
    # f is the nonlinear function
    # max number of times to split: maxlvl
    for mxl in range(maxlvl):
        h = np.sqrt(3)
        Ncmop = gmm.Ncomp
        G = uqgmmbase.GMM(None,None,None,gmm.currt)
        for i in range(Ncmop):
            fmu = f(gmm.m(i))
            S = sclnalg.sqrtm(gmm.P(i))
            invS = nplnalg.inv(S)
            phi=[0]*S.shape[1]
            for j in range(S.shape[1]):
                avec = S[:,j]
                avecnorm = avec/nplnalg.norm(avec)
                avecnormMag = 1/nplnalg.norm(np.matmul(invS,avecnorm))
                phi[j] = ( f(gmm.m(i)+h*avecnormMag*avecnorm)+f(gmm.m(i)-h*avecnormMag*avecnorm)-2*fmu )/(2*h**2)
            
            phinorms = [nplnalg.norm(phi[e]) for e in range(S.shape[1])]
            # print(phinorms)
            j=np.argmax(phinorms)
            avec = S[:,j]
            gmmcomp=uqsplit.splitGaussianND_principleAxis_ryanruss(avec,gmm.m(i),gmm.P(i),N,ruleoption)
            gmmcomp.currt = gmm.currt
            gmmcomp.scaleWt(gmm.w(i))
            G.appendGMM(gmmcomp)
        
        gmm = G
    
    return gmm

