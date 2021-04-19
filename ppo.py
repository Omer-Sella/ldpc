# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:19:32 2021

@author: Omer Sella
"""

import models
import utilityFunctions
import numpy as np
import scipy.signal
import torch
from mpiFunctions import mpi_statistics_scalar

OBSERVATION_DATA_TYPE = np.float32
INTERNAL_ACTION_DATA_TYPE = np.float32

epochs = 5
numberOfStepsPerEpoch = 10
seed = 7134066
localRandom = np.random.RandomState(seed)
maximumEpisodeLength = 3







class ppoBuffer:
    
    def __init__(self, observationDimension, internalActionDimensions, size, gamma = 0.99, lam = 0.95):
        self.observationBuffer = np.zeros((size , observationDimension), dtype = OBSERVATION_DATA_TYPE)
        self.nextObservationBuffer = np.zeros((size , observationDimension), dtype = OBSERVATION_DATA_TYPE)
        self.actionBuffer = np.zeros((size , internalActionDimensions), dtype = INTERNAL_ACTION_DATA_TYPE)
        self.advantageBuffer = np.zeros(size, dtype = np.float32)
        self.rewardBuffer = np.zeros(size, dtype = np.float32)
        self.returnBuffer = np.zeros(size, dtype = np.float32)
        self.valueBuffer = np.zeros(size, dtype = np.float32)
        self.logProbabilityBuffer = np.zeros(size, dtype = np.float32)
        self.gamma = gamma
        self.lam = lam
        self.counter = 0
        self. pathStartIndex = 0
        self.maximalSize = size
    
    def store(self, observation, action, reward, value, logProbability, nextObservation):
        assert self.counter < self.maximalSize
        self.observationBuffer[self.counter] = observation
        self.actionBuffer[self.counter] = action
        self.rewardBuffer[self.counter] = reward
        self.valueBuffer[self.counter] = value
        self.logProbabilityBuffer[self.counter] = logProbability
        self.nextObservationBuffer[self.counter] = nextObservation
        self.counter = self.counter + 1
            
    def finishPath(self, lastValue = 0):
        
        pathSlice = slice(self.pathStartIndex, self.counter)
        rewards = np.append(self.rewardBuffer[pathSlice], lastValue)
        values = np.append(self.valueBuffer[pathSlice], lastValue)
        
        # Generalise Advantage Estimation (GAE) lambda advantage calculation
        deltas = rewards[: -1] + self.gamma * values[1:] - values[: -1]
        self.advantageBuffer[pathSlice] = scipy.signal.lfilter([1], [1, float(-(self.gamma * self.lam))], deltas[::-1], axis=0)[::-1]
        self.returnBuffer[pathSlice] = scipy.signal.lfilter([1], [1, float(-(self.gamma))], rewards[::-1], axis=0)[::-1]
        
        self.pathStartIndex = self.counter
        
    def get(self):
        # Buffer has to be full before get. 
        # Omer Sella: why ?
        assert self.counter == self.maximalSize
        self.counter = 0
        self.pathStartIndex = 0
        advantageMean, advantageStd = mpi_statistics_scalar(self.advantageBuffer)
        self.advantageBuffer = (self.advantageBuffer - advantageMean) / advantageStd
        data = dict(observations = self.observationBuffer, nextObservations = self.nextObservationBuffer, 
                    actions = self.actionBuffer, returns = self.returnBuffer, advantage = self.advantageBuffer,
                    logProbabilities = self.logProbabilityBuffer)
        return {k: torch.as_tensor(v, dtype = torch.float32) for k,v in data.items()}

def ppo():
    
    
    
    
    
    # Initialise actor-critic
    myActorCritic = models.actorCritic(int, 2048, int, 16, 7, [64,64] , 'cpu')
    
    #
    
    for epoch in range(epochs):
        for t in range(numberOfStepsPerEpoch):
            vector = np.zeros(511, dtype = int)
            i, j, k, logpI, logpJ, logpK = myActorCritic.step(torch.as_tensor(observation))
            
            ## I don't have a critic to give value estimation yet, so set value to 1
            value = 1
            
            logProbabilityList = [logpI, logpJ, logpK]
            ## Omer Sella: temporarily generate a random sparse vector
            xCoordinate = utilityFunctions.numToBits(i, 1)
            yCoordinate = utilityFunctions.numToBits(j, 4)
            hotBits = localRandom.choice(511, k, replace = False)
            newVector[hotBits] = 1
            nextObservation, reward, done, _ = env.step(i, j, newVector)
            
            ## book keeping
            episodeReturn = episodeReturn + reward
            episodeLength = episodeLength + 1
            
            ## save to buffer and log
            ppoBuffer.store(observation, action, reward, logProbabilityList)
            logger.store(value)
            
            # Update observation to be nextObservation
            observation = nextObservation
            
            timeOut = (episodeLength == maximumEpisodeLength)
            terminate = (done or timeOut)
            epochEnding = (t == numberOfStepsPerEpoch - 1)
            
            if terminate or epochEnding:
                if timeOut or epochEnding:
                    # Get value estimation
                    value = 2
                else:
                    value = 0
                    
        # Save the model accordingly
        #if (epoch % saveModelFrequency == 0) or (epoch == (epochs - 1)):
            
        #perform ppo policy update
            
                