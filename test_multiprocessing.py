#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 18:56:59 2021

@author: na0043
"""
import multiprocessing as mp
import random
import time
def worker(name: str) -> None:
    print(f'Started worker {name}')
    worker_time = random.choice(range(1, 5))
    time.sleep(worker_time)
    print(f'{name} worker finished in {worker_time} seconds')

def ff():
    # mp.set_start_method('spawn', force=True)
    # ctx = mp.get_context('spawn')
    # lock = ctx.Lock()
    # seqQ = ctx.Queue()
    # resQ = ctx.Queue()
    # ExitFlag = ctx.Event()
    # ExitFlag.clear()
    # try:
    #     mp.set_start_method('fork')
    # except RuntimeError:
    #     pass
    
    processes = []
    for i in range(5):
        # lock.acquire()
        process = mp.Process(target=worker, 
                                          args=(f'computer_{i}',))
        processes.append(process)
        process.start()
        # lock.release()
    
    for proc in processes:
        proc.join()

ff()