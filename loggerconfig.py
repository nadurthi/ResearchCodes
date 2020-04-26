#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:17:27 2020

@author: nagnanamus
"""


import logging
from logging.config import dictConfig

# adding custom level
logging.VERBOSE = 5
# logging._levelToName.update({
#     5: 'VERBOSE',
#     'VERBOSE': 5,
# })
logging.addLevelName(logging.VERBOSE, "VERBOSE")

# method for custom level


def verbose(self, msg, *args, **kwargs):
    if self.isEnabledFor(logging.VERBOSE):
        self._log(logging.VERBOSE, msg, args, **kwargs)


logging.Logger.verbose = verbose


dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - [%(levelname)s] %(name)s [%(module)s.%(funcName)s:%(lineno)d]: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        }
    },
    'handlers': {
        'default': {
            'level': 'VERBOSE',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        }
    },
    'loggers': {
        '__main__': {  # logging from this module will be logged in VERBOSE level
            'handlers': ['default'],
            'level': 'VERBOSE',
            'propagate': False,
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['default']
    },
})
