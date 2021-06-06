import numpy as np
import random 

def samplePMF(pmf,Ns):
    # pmf=[p1,p2,p3...pN]
    # Ns is number of samples
    # return index
    states = np.arange(len(pmf))
    return random.choices(states,weights=pmf,k=Ns)


