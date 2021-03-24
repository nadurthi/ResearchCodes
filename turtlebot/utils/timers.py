import time
from contextlib import ContextDecorator

class TimingContext(object):

    def __init__(self,msg="Code",printit=True):
        self.st = time.time()
        self.msg = msg
        self.printit = printit
    def __enter__(self):
        self.st = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.et = time.time()
        if self.printit:
            print(self.msg+" Time taken = ",self.et-self.st)
        
        


class mycontext(ContextDecorator):
    def __enter__(self):
        print('Starting')
        return self

    def __exit__(self, *exc):
        print('Finishing')
        return False

@mycontext()
def function():
    print('The bit in the middle')