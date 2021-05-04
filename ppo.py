# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:19:32 2021

@author: Omer Sella
"""
import numpy as np
import torch
from torch.optim import Adam

import models
import utilityFunctions
import scipy.signal
from mpiFunctions import mpi_statistics_scalar, num_procs
# Needed for the update / training function
from mpiFunctions import mpi_avg
from mpiFunctions import mpi_avg_grads
import time
import gym

OBSERVATION_DATA_TYPE = np.float32
INTERNAL_ACTION_DATA_TYPE = np.float32

epochs = 5
numberOfStepsPerEpoch = 10
seed = 7134066
localRandom = np.random.RandomState(seed)
maximumEpisodeLength = 3
clipRatio = 0.2
policyLearningRate = 3e-4
valueFunctionLearningRate = 1e-3
loggerKeyWords = ['Episode return', 'Episode length']
policyTrainIterations = 80
targetKL = 1.5 * 0.01
valueFunctionTrainIterations = 80
loggerPath = utilityFunctions.PROJECT_PATH + "/temp/"
MAXIMUM_NUMBER_OF_HOT_BITS = 5
INTERNAL_ACTION_SPACE_SIZE = 1 + 1 + 1 + MAXIMUM_NUMBER_OF_HOT_BITS
SAVE_MODEL_FREQUENCY = 10


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
                    actions = self.actionBuffer, returns = self.returnBuffer, advantages = self.advantageBuffer,
                    logProbabilities = self.logProbabilityBuffer)
        return {k: torch.as_tensor(v, dtype = torch.float32) for k,v in data.items()}




def ppo(environmentFunction):
  
    
    # localStepsPerEpochs is the number of steps each MPI process spends in an epoch.
    localStepsPerEpoch = int( numberOfStepsPerEpoch / num_procs() )
    
    # Initialise actor-critic
    myActorCritic = models.actorCritic(int, 2048, int, 16, INTERNAL_ACTION_SPACE_SIZE, [64,64] , 'cpu')
    
    policyOptimizer = Adam(myActorCritic.policy.parameters(), policyLearningRate)
    valueFunctionOptimizer = Adam(myActorCritic.value.parameters(), valueFunctionLearningRate)
    
    myLog = utilityFunctions.logger(loggerKeyWords, loggerPath, fileName = 'experiment.txt')
    #myLog.setupPytorchSave(myActorCritic)
    
    myBuffer = ppoBuffer(2048, INTERNAL_ACTION_SPACE_SIZE, localStepsPerEpoch)
    
    # Internal function to ppo, so exposed to all parameters passed to ppo
    def computeLoss(data):
        #clipRatio is a (hyper)parameter passed to ppo
        observations = data['observations']
        actions = data['actions']
        advantages = data['advantages']
        logProbabilitiesOld = data['logProbabilities']
        returns = data['returns']
        
        # Omer Sella: is policy(pi) the policy model parameters here ?
        policy, logProbabilities = myActorCritic.policy(observations, actions)
        ratio = torch.exp(logProbabilities - logProbabilitiesOld)
        advantagesClipped = torch.clamp(ratio, 1 - clipRatio, 1 + clipRatio) * advantages
        policyLoss = -1 * ( torch.min(ratio * advantages, advantagesClipped)).mean()
        
        approximatedKL = (logProbabilitiesOld - logProbabilities).mean().item()
        entropy = policy.entropy().mean().item()
        clippedPart = ratio.gt(1 + clipRatio) | ratio.lt(1-clipRatio)
        clippedFraction = torch.as_tensor(clippedPart, dtype = torch.float32).mean().item()
        policyInfo = dict(kl = approximatedKL, entropy = entropy, clippedFraction = clippedFraction)
        
        valueLoss = ( (myActorCritic.value(observations) - returns) ** 2).mean()
        
        return policyLoss, policyInfo, valueLoss
    
    def update():
        data = myBuffer.get()
        policyLossOld, policyInfoOld, valueLossOld = computeLoss(data)
        policyLossOld = policyLossOld.item()
        valueLossOld = valueLossOld.item()
        
        # Policy training
        for i in range(policyTrainIterations):
            policyOptimizer.zero_grad()
            lossPolicy, policyInformation, _ = computeLoss(data)
            klAverage = mpi_avg(policyInformation['kl'])
            if klAverage > targetKL:
                myLog.logPrint('Early stopping at step %d due to reaching maximal KL divergence.' %i)
                break
            lossPolicy.backwards()
            mpi_avg_grads(myActorCritic.policy)
            policyOptimizer.step()
        
        # Omer Sella: not implemented: myLog.store(stopIteration = i)
    
        # Value function training
        for i in range(valueFunctionTrainIterations):
            valueFunctionOptimizer.zero_grad()
            _, _, lossValue = computeLoss(data)
            lossValue.backwards()
            mpi_avg_grads(myActorCritic.valueFunction)
            valueFunctionOptimizer.step()
            
        # Omer Sella: not implemented: log changes due to update.
        # log.store policyLossOld, valueLossOld, kl, entropy, clipFraction, delta between lossPolicy and policyLossOld, delta between lossValue and valueLossOld
    
    #
    
    startTime = time.time()
    observation = environmentFunction.reset()
    episodeReturn = 0
    episodeLength = 0
    
    for epoch in range(epochs):
        for t in range(localStepsPerEpoch):
            newVector = np.zeros(511, dtype = int)
            i, j, k, coordinates, logpI, logpJ, logpK, logpC = myActorCritic.step(torch.as_tensor(observation))
            
            ## I don't have a critic to give value estimation yet, so set value to 1
            value = 1
            
            logProbabilityList = [logpI, logpJ, logpK, logpC]
            ## Omer Sella: temporarily generate a random sparse vector
            xCoordinate = utilityFunctions.numToBits(i, 1)
            yCoordinate = utilityFunctions.numToBits(j, 4)
            newVector[coordinates[0:k]] = 1
            action = np.hstack((np.hstack((xCoordinate, yCoordinate)), newVector))
            nextObservation, reward, done, _ = environmentFunction.step(i, j, newVector)
            
            ## book keeping
            episodeReturn = episodeReturn + reward
            episodeLength = episodeLength + 1
            
            ## save to buffer and log
            ppoBuffer.store(observation, action, reward, logProbabilityList, nextObservation)
            myLog.keyValue('value', value)
            
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
                myBuffer.finishPath()
                if terminate:
                    myLog.keyValue('Episode return', episodeReturn)
                    myLog.keyValue('Episode length', episodeLength)
                    observation = environmentFunction.reset()
                    episodeLength = 0
                    episodeReturn = 0
                    
        # Save the model accordingly
        if (epoch % SAVE_MODEL_FREQUENCY == 0) or (epoch == (epochs - 1)):
        
            # Omer Sella: not yet implemented: this is a state savingof the experiment. I need to add this ability to the logger.
            #myLog.save_state
            pass    
        
        # ppo policy update
        update()
        
        
if __name__ == '__main__':
    
    ppo(lambda : gym.make('gym_ldpc:ldpc-v0'))      
            
                