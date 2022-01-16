# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 17:01:44 2021

@author: Nagnanamus
"""
import multiprocessing as mp
import queue
import time


def consumer(Func,ExitFlag,inQ,outQ):
    # print("in consumer")
    while(True):
        inputarg = None
        try:
            inputarg = inQ.get(True,0.2)
        except queue.Empty:
            inputarg = None
        
        if inputarg is not None:
            
            # print("inputarg = ",inputarg)
            result=Func(inputarg)
            outQ.put(result)
            
            time.sleep(0.1)
        
        if (ExitFlag.is_set() and inQ.empty()):
            # print("exiting consumer")
            break
        
class ParallelConsumer():
    def __init__(self,Fconsumer,Nproc=4,maxInQ=10):
        self.Nproc=Nproc
        self.Fconsumer=Fconsumer
        self.maxInQ=maxInQ
        self.inQ = mp.Queue(maxInQ) # mi took the max size of queue as 50 here
        self.outQ = mp.Queue()
        self.ExitFlag = mp.Event()
        self.ExitFlag.clear()
        self.cnt=0
        self.processes = []
    
        for i in range(Nproc):
            p = mp.Process(target=consumer, args=(Fconsumer,self.ExitFlag,self.inQ,self.outQ))
            p.start()
            self.processes.append( p )
            time.sleep(0.1)
    
    def pushInputArg(self,inputarg):
        self.inQ.put(inputarg)
        self.cnt+=1
    
    def iterateOutput(self):
        i=0
        while True:
            res=None
            try:
                res=self.outQ.get(True,0.1)
            except queue.Empty:
                time.sleep(0.2)
            
            if res is not None:
                yield res
                i+=1
                
            if self.inQ.empty() and self.outQ.empty() and self.cnt==i:
                break
            
        self.ExitFlag.set()
        self.finish()
        
    def finish(self):
        for p in self.processes:
            p.join()
            
        
if __name__=="__main__":
    def Fconsumer(inputarg):
        i,line = inputarg
        # print("Fconsumer = ",i,len(line))
        
        return (i,len(line))
    
    pc=ParallelConsumer(Fconsumer,Nproc=2,maxInQ=10)
    ss=['aa','bbbb','cdcdsgfsdg','sfe']
    for i,s in enumerate(ss):
        pc.pushInputArg((i,s))
    result=[]
    for res in pc.iterateOutput():
        # print("output res = ",res)
        result.append(res)
    
    print("result = ")
    print(result)