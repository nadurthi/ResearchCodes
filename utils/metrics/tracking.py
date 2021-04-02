# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def multiTarget_tracking_metrics(simmanager,groundtargetset,targetset):
    # tracking error for active times
    for i in range(groundtargetset.ntargs):
        if targetset[i].isSearchTarget():
            continue
        
                