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
            layerList = layerList + [nn.Linear(lengths[l], lengths[l + 1]), activation()]
        self.layers = nn.ModuleList(layerList)
        self.outputDimension = outputLength
        
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    
    
class actorCritic(nn.Module):
    def __init__(self, observationSpaceType, observationSpaceSize, actionSpaceType, 
                 actionSpaceSize, hiddenEncoderSize, maximumNumberOfHotBits, hiddenLayerParameters, actorCriticDevice = 'cpu'):
        super().__init__()
        self.observationSpaceType = observationSpaceType
        self.observationSpaceSize = observationSpaceSize
        self.actionSpaceType = actionSpaceType
        self.actionSpaceSize = actionSpaceSize
        self.device = actorCriticDevice
        self.maximumNumberOfHotBits = maximumNumberOfHotBits
        self.hiddenEncoderSize = hiddenEncoderSize
        
        self.rowCoordinateRange = 2
        self.columnCoordinateRange = 16
        self.circulantSize = 511
        self.defaultHiddenLayerSizes = [64]
        self.defaultActivation = nn.Identity

        self.encoder = explicitMLP(observationSpaceSize, hiddenEncoderSize, [hiddenEncoderSize, hiddenEncoderSize])

        self.rowCoordinateModel = explicitMLP(self.hiddenEncoderSize, self.rowCoordinateRange, self.defaultHiddenLayerSizes)

        self.columnCoordinateModel = explicitMLP(self.hiddenEncoderSize + 1, self.columnCoordinateRange, self.defaultHiddenLayerSizes)
        
        self.numberOfHotBitsModel = explicitMLP(self.hiddenEncoderSize + 2, self.maximumNumberOfHotBits, self.defaultHiddenLayerSizes)
        
        self.kHotVectorGenerator = explicitMLP(self.hiddenEncoderSize + 3, self.circulantSize, self.defaultHiddenLayerSizes)
        
        self.encoder2 = explicitMLP(self.circulantSize, self.hiddenEncoderSize + 3, self.defaultHiddenLayerSizes)
        
        ## Omer: A note to self: the critic valuates the present state, not the state you are going to be in after taking the action.
        self.critic = explicitMLP(self.hiddenEncoderSize, 1, [hiddenEncoderSize, hiddenEncoderSize])
        self.to(actorCriticDevice)
    
    
    
    def actorActionToEnvAction(self, actorAction):
        i, j, k, hotCoordinates = actorAction
        """
        The actor is expected to produce i, j, and up to k coordinates which will be hot.
        The environment is expecting i,j and a binary vector.
        """
        binaryVector = np.zeros(self.circulantSize, dtype = MODELS_BOOLEAN_TYPE)
        binaryVector[hotCoordinates[0:k]] = 1
        environmentStyleAction = [i, j, binaryVector]
        return environmentStyleAction
    
    def step(self, observations, action = None):
        
        # The step function has 3 modes:
        #   1. Training - this is where we use the model and sample from the distributions parametrised by the model.
        #   2. Most probable action - this is where we use the deterministic model and instead of sampling take the most probable action.
        #   3. Both observations AND actions are provided, in which case we are evaluating log probabilities
        
        if action is not None:
            action = torch.as_tensor(action, device = self.device)
        
        encodedObservation = self.encoder(observations)
        logitsForIChooser = self.rowCoordinateModel(encodedObservation)
        iCategoricalDistribution = Categorical(logits = logitsForIChooser)
        
        if action is not None:
            i = action[:, 0]
        elif self.training:
            i = iCategoricalDistribution.sample()
        else:
            i = torch.argmax(logitsForIChooser)
                
        
                
        # Omer Sella: now we need to append i to the observations
        ## Omer Sella: when acting you need to sample and concat. When evaluating, you need to break the action into internal components and set to them.
        ## Then log probabilities are evaluated at the end (regardless of whether this was sampled or given)
        
        iAppendedObservations = torch.cat([encodedObservation, i], dim = -1)
        logitsForJChooser = self.columnCoordinateModel(iAppendedObservations)
        jCategoricalDistribution = Categorical(logits = logitsForJChooser)
        
        if action is not None:
            j = action[:, 1]
        elif self.training:    
            j = jCategoricalDistribution.sample()
        else:
            j = torch.argmax(logitsForJChooser)
                
        # Omer Sella: now we need to append j to the observations
        jAppendedObservations = torch.cat([iAppendedObservations, i], dim = -1)
        logitsForKChooser = self.numberOfHotBitsModel(jAppendedObservations)
        kCategoricalDistribution = Categorical(logits = logitsForKChooser)
        
        if action is not None:
            k = action[:, 2]
        elif self.training:
            k = kCategoricalDistribution.sample()
        else:
            k = torch.argmax(logitsForKChooser)
                
        kAppendedObservations = torch.cat([jAppendedObservations, k], dim = -1)
        setEncodedStuff = self.encoder2(kAppendedObservations)
        logProbCoordinates = np.zeros(self.maximumNumberOfHotBits)
        
        if action is not None:
            coordinates = action[:, 3 : 3 + self.maximumNumberOfHotBits]
            idx = 0
            while idx < k:
                logitsForCoordinateChooser = self.kHotVectorGenerator(setEncodedStuff)
                circulantSizeCategoricalDistribution = Categorical(logits = logitsForCoordinateChooser)
                newCoordinate = coordinates[idx]
                logProbCoordinates[idx] = circulantSizeCategoricalDistribution.log_prob(newCoordinate)
                setEncodedStuff = setEncodedStuff + logitsForCoordinateChooser
        elif self.training:
            coordinates = -1 * np.ones(self.maximumNumberOfHotBits)
            idx = 0
            while idx < k:
                logitsForCoordinateChooser = self.kHotVectorGenerator(setEncodedStuff)
                circulantSizeCategoricalDistribution = Categorical(logits = logitsForCoordinateChooser)
                newCoordinate = circulantSizeCategoricalDistribution.sample()
                coordinates[idx] = newCoordinate
                logProbCoordinates[idx] = circulantSizeCategoricalDistribution.log_prob(newCoordinate)
                setEncodedStuff = setEncodedStuff + logitsForCoordinateChooser
        else:
            coordinates = -1 * np.ones(self.maximumNumberOfHotBits)
            idx = 0
            while idx < k:
                logitsForCoordinateChooser = self.kHotVectorGenerator(setEncodedStuff)
                circulantSizeCategoricalDistribution = Categorical(logits = logitsForCoordinateChooser)
                newCoordinate = torch.argmax(logitsForCoordinateChooser)
                coordinates[idx] = newCoordinate
                logProbCoordinates[idx] = circulantSizeCategoricalDistribution.log_prob(newCoordinate)
                setEncodedStuff = setEncodedStuff + logitsForCoordinateChooser
                
                    
                #log probs
        logpI = iCategoricalDistribution.log_prob(i).sum(axis = -1)                    
        logpJ = jCategoricalDistribution.log_prob(j).sum(axis = -1)
        logpK = kCategoricalDistribution.log_prob(k).sum(axis = -1)
        
                
                
        return i, j, k, coordinates, logpI, logpJ, logpK, logProbCoordinates

def testExplicitMLP():
    inputLength = 2048
    outputLength = 16
    hiddenLayersLengths = [64, 64]
    myMLP = explicitMLP(inputLength, outputLength, hiddenLayersLengths)
    return myMLP

def testExplicitMLPForward():
    myMLP = testExplicitMLP()
    testVector = torch.rand(2048)
    y = myMLP(testVector)
    return y

