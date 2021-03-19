# -*- coding: utf-8 -*-
import multiprocessing as mp
import time

import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


class ProcessPlotter(object):
    def __init__(self):
        self.x = []
        self.y = []
        print("process init")
    def terminate(self):
        plt.close('all')

    def call_back(self):
        print("pollign start")
        while self.pipe.poll():
            print("pollign")
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                self.x.append(command[0])
                self.y.append(command[1])
                self.ax.plot(self.x, self.y, 'ro')
        self.fig.canvas.draw()
        return True

    def __call__(self, pipe):
        print('starting plotter...')

        self.pipe = pipe
        self.fig, self.ax = plt.subplots()
        timer = self.fig.canvas.new_timer(interval=1000)
        timer.add_callback(self.call_back)
        timer.start()

        print('...done')
        plt.show()
        
        
        
def gg(pipe):
    print('starting gg')
    pass
        
        
        
        
        
class NBPlot(object):
    def __init__(self):
        self.plot_pipe, plotter_pipe = mp.Pipe()
        plotter = ProcessPlotter()
        self.plot_process = mp.Process(
            target=gg, args=(plotter_pipe,), daemon=None)
        self.plot_process.start()
        print("joining")
        self.plot_process.join()
        print("joining done")
        self.plotter_pipe=plotter_pipe
    def plot(self, finished=False):  
        
        if finished:
            self.plot_pipe.send(None)
        else:
            data = np.random.random(2)
            self.plot_pipe.send(data)
            # time.sleep(1)


def main():
    pl = NBPlot()
    for ii in range(10):
        pl.plot()
        time.sleep(0.5)
    pl.plot(finished=True)


if __name__ == '__main__':
    # mp.set_start_method("forkserver")
    mp.freeze_support()
    plot_process = mp.Process(
            target=gg, args=(22,), daemon=None)
    plot_process.start()
    plot_process.join()
    
    pl = NBPlot()
    for ii in range(10):
        pl.plot()
        time.sleep(0.5)
    pl.plot(finished=True)