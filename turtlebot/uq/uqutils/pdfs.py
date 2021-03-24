#!/usr/bin/env python
"""
Documentation for this imm module

More details.
"""

import numpy as np
import numpy.linalg as npalg

def isInNsig(x,m,P,N):
    return np.matmul(np.matmul((x-m),npalg.inv(P) ) , (x-m)) <= N**2