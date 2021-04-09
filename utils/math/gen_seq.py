# -*- coding: utf-8 -*-

import numpy as np
import numba as nb

def independent_seq_generator(Xsets):
    """
    Xsets = [Xset1,Xset2,Xset3]
    seqL = 3
    chose one from each and build the seq
    """
    seqL = len(Xsets)
    cntrs = np.zeros(seqL,dtype=np.int)
    while True:
        Xseq=[]
        for i in range(seqL):
            Xseq.append( Xsets[i][cntrs[i]] )
            if i==seqL-1:
                cntrs[i] = cntrs[i]+1
        
        for i in range(seqL-1,0,-1):
            if cntrs[i] == len(Xsets[i]):
                cntrs[i]=0
                cntrs[i-1] = cntrs[i-1] + 1
        yield Xseq
        if cntrs[0] == len(Xsets[0]):
            break
        
def dependent_seq_generator(seqL,ind0,Xset,Xconst):
    """
    seqL is the length of the required sequence
    Xset=[0,1,2,3..]  NO NEGATIVE NUMBERS!!!
    Xconst={0:[1,2,3,4],
            1:[0,2,3]]
    first col is the node
    rest of the cols are  accessible nodes from first col    
    Parameters
    ----------
    ind0 : start node in Xseq
        DESCRIPTION.
    Xset : List or array of ALL the nodes
        DESCRIPTION.
    Xconst : Constraints
    Returns
    -------
    None.

    """
    cntrs = np.zeros(len(Xset))
    Xconst_ctr={s:{} for s in Xset}
    brkflg=0
    while True:
        Xseq=[ind0]
        for i in range(1,seqL):
            prevnode = Xseq[i-1]
            nextset = Xconst[prevnode]
            if i not in Xconst_ctr[prevnode]:
                Xconst_ctr[prevnode][i] = 0
                
            cnode = Xconst_ctr[prevnode][i]
            Xseq.append( nextset[cnode] )
            if i==seqL-1:
                Xconst_ctr[prevnode][i] = Xconst_ctr[prevnode][i] + 1
            
        for i in range(seqL-1,0,-1):
            prevnode = Xseq[i-1]
            nextset = Xconst[prevnode]
            if Xconst_ctr[prevnode][i] == len(nextset):
                Xconst_ctr[prevnode][i] = 0
                j=i-1
                if j==0:
                    brkflg=1
                else:
                    prevnode = Xseq[j-1]
                    Xconst_ctr[prevnode][j] = Xconst_ctr[prevnode][j] + 1
        
        yield Xseq
        # 0th node is fixed
        # prevnode = Xseq[0]
        # nextset = Xconst[prevnode]    
        if brkflg:
            break
            
            
if __name__=="__main__":
    ind0 = 10
    seqL = 3
    Xset = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    Xconst={ 10: [11,12,13,14],
             11: [10,15,16,17],
             12: [18,16,19,10],
             13: [10,17,20,21],
             14: [10,19,22,21],
             15: [11],
             16: [11,12],
             17: [11,13],
             18: [12],
             19: [12,14,23],
             20: [13],
             21: [14,13,24],
             22: [14,23,24],
             23: [19,22],
             24: [21,22],
             
        }
    
    for Xseq in dependent_seq_generator(seqL,ind0,Xset,Xconst):
        print(Xseq)
        # gogo = input("next")
               
    print("independent sequences")

    Xsets=[[1,2,3],[4,5,6],[7,8]]
    for Xseq in independent_seq_generator(Xsets):
        print(Xseq)
