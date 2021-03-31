# -*- coding: utf-8 -*-

from physmodels import motionmodels as phymm



class TransitionModel(phymm.MotionModel):
    motionModelName = 'TransitionModel'
    def __init__(self,P,X):
        super().__init__()
        self.X = X
        self.P = P
        self.fn = X.shape[0]

		

	

	
