"""
Created on Fri Apr 24 11:17:27 2020

@author: nagnanamus
"""


import logging
from logging.config import dictConfig
import os

logging.VERBOSE = 5
logging.addLevelName(logging.VERBOSE, "VERBOSE")
def verbose(self, msg, *args, **kwargs):
    if self.isEnabledFor(logging.VERBOSE):
        self._log(logging.VERBOSE, msg, args, **kwargs)

logging.TIMING = 6
logging.addLevelName(logging.TIMING, "TIMING")
def timing(self, msg, *args, **kwargs):
    if self.isEnabledFor(logging.TIMING):
        self._log(logging.TIMING, msg, args, **kwargs)
        

logging.Logger.verbose = verbose
logging.Logger.timing = timing

class TimingFilter(logging.Filter):
    def __init__(self,*args,**kwargs):
        pass
    
    def filter(self,logRecord):
        return logRecord.levelno == 6

class TimingFormatter(logging.Formatter):
    def format(self, record):
        record.funcName = record.args.get("funcName")
        record.timeTaken = record.args.get("timeTaken")
        record.funcArgstr = record.args.get("funcArgstr")
        return super().format(record)
    
    
dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - [%(levelname)s] %(name)s [%(module)s.%(funcName)s:%(lineno)d]: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'timeformatter': {
            'class': 'loggerconfig.TimingFormatter',
            'format': '%(asctime)s - [%(levelname)s] %(name)s [%(process)d,%(thread)d,%(module)s,%(funcName)s,%(lineno)d]# %(message)s # %(funcName)s # %(funcArgstr)s # %(timeTaken)f',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        }
    },
    'filters': {
        'timingfilter': {
            '()': TimingFilter,
            'param': 'noshow',
        }
    },
    'handlers': {
        'default': {
            'level': 'VERBOSE',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        },
        'file': {
            'level': 'VERBOSE',
            'class': 'logging.FileHandler',
            'filename': 'junk.log',
            'mode': 'a',
            'formatter': 'standard',
        },
        'timefile': {
            'level': 'TIMING',
            'class': 'logging.FileHandler',
            'filename': 'junktime.log',
            'mode': 'a',
            'formatter': 'timeformatter',
            'filters' : ['timingfilter']
        }
        
    },
    'loggers': {
        '': {  # logging from this module will be logged in VERBOSE level
            'handlers': ['file','timefile'], #'default',
            'level': 'VERBOSE',
            'propagate': False,
        },
        '__main__': {  # logging from this module will be logged in VERBOSE level
            'handlers': ['file','timefile'],
            'level': 'VERBOSE',
            'propagate': False,
        },
    },
    # 'root': {
    #     'level': 'VERBOSE',
    #     'handlers': ['default','file','timefile']
    # },
    
})

def getLogger(name):
    logger = logging.getLogger(name)
    return logger



