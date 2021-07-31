# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 17:01:44 2021

@author: Nagnanamus
"""
import multiprocessing as mp
import queue
import time

def consumer(ExitFlag,inQ,outQ):
    while(True):
        lines = None
        try:
            lines = inQ.get(True,0.2)
        except queue.Empty:
            lines = None
        
        if lines is not None:
            # do some work
            result = 1
            outQ.put(result)
            print("Working on ",lines)
            time.sleep(1)
            pass
        
        if (ExitFlag.is_set() and inQ.empty()):
            print("exiting producer")
            break
        


inQ = mp.Queue(5) # mi took the max size of queue as 50 here
outQ = mp.Queue()
ExitFlag = mp.Event()
ExitFlag.clear()

processes = []
Ncore = 4
for i in range(Ncore):
    p = mp.Process(target=consumer, args=(ExitFlag,inQ,outQ))
    processes.append( p )
    p.start()

time.sleep(3)

if __name__=="__main__":
    LinesSets=[1,2,3,4,5,6,7,8,9,10]
    for lines in LinesSets:
        if not inQ.full():
            inQ.put(lines)
        else:
            print("queue full so producer is waiting")
        
        
        
        
        
        
    while True:
        if inQ.empty():
            break
        else:
            time.sleep(5)
            
    ExitFlag.set()
    time.sleep(5)
    
    for p in processes:
        p.join()
        