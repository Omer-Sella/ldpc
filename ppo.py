# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:19:32 2021

@author: Omer Sella
"""

import models
import utilityFunctions
import numpy as np

epochs = 5
numberOfStepsPerEpoch = 10
seed = 7134066
localRandom = np.random.RandomState(seed)
maximumEpisodeLength = 3

def ppo():
    
    myActorCritic = models.actorCritic(int, 2048, int, 16, 7, [64,64] , 'cpu')
    
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
                