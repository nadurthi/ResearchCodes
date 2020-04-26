# -*- coding: utf-8 -*-


import uq.quadratures.cubatures as uqcb
from loggerconfig import *
import numpy as np
import uq.motfilter.mot as uqmot


logger = logging.getLogger(__name__)

logger.info('Info log message')
logger.debug('debug message')
uqmot.foo()
logger.error('error example')
logger.verbose('verbose log message')
# try:
#     raise Exception('exception message')
# except:
#     logger.exception('error occured')


logger.debug('debug message')
logger.info('info message')
logger.warning('warn message')
logger.error('error message')
logger.critical('critical message')
# %%


mu = np.zeros(3)
P = np.eye(3)
X, w = uqcb.UT_sigmapoints(mu, P)

X, w = uqcb.CUT4pts_gaussian(mu, P)
