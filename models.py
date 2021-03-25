# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:19:52 2021

@author: Omer Sella
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

MODELS_BOOLEAN_TYPE = np.bool
MODELS_INTEGER_TYPE = np.int32


class explicitMLP(nn.Module):
        
    

    """
    explicitMLP creates a multi layer perceptron with explicit input and output lengths.
    if hiddenLayersLengths is not an empty list it will create hidden layers with the specified lengths as input lengths.
    default activation is the identity.
    """
    def __init__(self, inputLength, outputLength, hiddenLayersLengths, intermediateActivation = nn.Identity, outputActivation = nn.Identity):
        super().__init__()
        lengths = [inputLength] + hiddenLayersLengths + [outputLength]
        self.outputActivation = outputActivation
        layerList = []
        
        for l in range(len(lengths) - 1):
            if (l < (len(lengths) - 2)):
                activation = intermediateActivation
            else:
                activation = outputActivation
            layerList = layerList + [nn.Linear(lengths[l], lengths[l+1]), activation()]
        self.layers = nn.ModuleList(layerList)
        self.outputDimension = outputLength
        
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    
    
class actorCritic(nn.Module):
    def __init__(self, observationSpaceType, observationSpaceSize, actionSpaceType, actionSpaceSize, maximumNumberOfHotBits, hiddenLayerParameters, actorCriticDevice):
        super().__init__()
        self.observationSpaceType = observationSpaceType
        self.observationSpaceSize = observationSpaceSize
        self.actionSpaceType = actionSpaceType
        self.actionSpaceSize = actionSpaceSize
        self.device = actorCriticDevice
        self.maximumNumberOfHotBits = maximumNumberOfHotBits
        self.rowCoordinateRange = 2
        self.columnCoordinateRange = 16
        self.circulantSize = 511
        self.defaultHiddenLayerSizes = [64,64]
        self.defaultActivation = nn.Identity

        self.rowCoordinateModel = explicitMLP(self.observationSpaceSize, self.rowCoordinateRange, self.defaultHiddenLayerSizes)

        self.columnCoordinateModel = explicitMLP(self.observationSpaceSize + 1, self.columnCoordinateRange, self.defaultHiddenLayerSizes)
        
        self.numberOfHotBitsModel = explicitMLP(self.observationSpaceSize + 2, self.maximumNumberOfHotBits, self.defaultHiddenLayerSizes)
        
        self.to(actorCriticDevice)
    
    
    
    def actorActionToEnvAction(self, actorAction):
        i, j, hotCoordinates = actorAction
        """
        The actor is expected to produce i, j, and up to k coordinates which will be hot.
        The environment is expecting i,j and a binary vector.
        """
        binaryVector = np.zeros(self.circulantSize, dtype = MODELS_BOOLEAN_TYPE)
        binaryVector[hotCoordinates] = 1
        environmentStyleAction = [i, j, binaryVector]
        return environmentStyleAction
    
    def step(self, observations):
        with torch.no_grad():
            iCategoricalDistribution = Categorical(logits = self.rowCoordinateModel(observations))
            i = iCategoricalDistribution.sample()
            logpI = iCategoricalDistribution.log_prob(i).sum(axis = -1)
            # Omer Sella: now we need to append i to the observations
            iAppendedObservations = torch.cat([observations, i], dim = -1)
            
            jCategoricalDistribution = self.columnCoordinateModel(iAppendedObservations)
            j = jCategoricalDistribution.sample()
            logpJ = jCategoricalDistribution.log_prob(j).sum(axis = -1)
            # Omer Sella: now we need to append j to the observations
            jAppendedObservations = torch.cat([iAppendedObservations, i], dim = -1)
            kCategoricalDistribution = self.numberOfHotBitsModel(jAppendedObservations)
            k = kCategoricalDistribution.sample()
            logpK = kCategoricalDistribution.log_prob(k).sum(axis = -1)
        return i, j, k, logpI, logpJ, logpK

def testExplicitMLP():
    inputLength = 2048
    outputLength = 16
    hiddenLayersLengths = [64, 64]
    myMLP = explicitMLP(inputLength, outputLength, hiddenLayersLengths)
    return myMLP

def testForward():
    myMLP = testExplicitMLP()
    testVector = torch.rand(2048)
    y = myMLP(testVector)
    return y

