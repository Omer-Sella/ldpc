# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 16:18:15 2021

@author: Omer Sella
"""

"""
This implements replay buffer for ppo.
It is similar to the code in openAI spinningup and I re-implemented it to understand it better.
"""

import numpy as np

OBSERVATION_DATA_TYPE = np.float32

class ppoBuffer:
    
    def __init__(self, observationDimension, internalActionDimensions, size, gamma = 0.99, lambda = 0.95):
        
        self.observationBuffer = np.zeros(observationDimension, dtype = OBSERVATION_DATA_TYPE)
        self.