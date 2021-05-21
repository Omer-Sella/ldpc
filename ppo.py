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
loggerKeyWords = ['value', 'Episode return', 'Episode length']
policyTrainIterations = 80
targetKL = 1.5 * 0.01
valueFunctionTrainIterations = 80
loggerPath = utilityFunctions.PROJECT_PATH + "/temp/"
MAXIMUM_NUMBER_OF_HOT_BITS = 5
INTERNAL_ACTION_SPACE_SIZE = 1 + 1 + 1 + MAXIMUM_NUMBER_OF_HOT_BITS
SAVE_MODEL_FREQUENCY = 10


class ppoBuffer:
    
    def __init__(self, observationDimension, internalActionDimensions, size, gamma = 0.99, lam = 0.95):
        self.observationBuffer = [] #np.zeros((size , observationDimension), dtype = OBSERVATION_DATA_TYPE)
        self.nextObservationBuffer = [] #= np.zeros((size , observationDimension), dtype = OBSERVATION_DATA_TYPE)
        self.actionBuffer = [] #np.zeros((size , internalActionDimensions), dtype = INTERNAL_ACTION_DATA_TYPE)
        self.advantageBuffer = [] #np.zeros(size, dtype = np.float32)
        self.rewardBuffer = [] #np.zeros(size, dtype = np.float32)
        self.returnBuffer = [] #np.zeros(size, dtype = np.float32)
        self.valueBuffer = [] #np.zeros(size, dtype = np.float32)
        self.logProbabilityBuffer = [] #np.zeros(size, dtype = np.float32)
        self.gamma = gamma
        self.lam = lam
        self.counter = 0
        self. pathStartIndex = 0
        self.maximalSize = size
    
    def store(self, observation, action, reward, value, logProbability, nextObservation):
        assert self.counter < self.maximalSize
        self.observationBuffer.append(observation)
        self.actionBuffer.append(action)
        self.rewardBuffer.append(reward)
        self.valueBuffer.append(value)
        self.logProbabilityBuffer.append(logProbability)
        self.nextObservationBuffer.append(nextObservation)
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


def computeLoss(actorCritic, data, clipRatio, valueFunctionCoefficients, entropyCoefficients, device = 'None'):
    #clipRatio is a (hyper)parameter passed to ppo
    observations = data['observations']
    actions = data['actions']
    advantages = data['advantages']
    logProbabilitiesOld = data['logProbabilities']
    returns = data['returns']
        
    # Omer Sella: is policy(pi) the policy model parameters here ?
    predictions = myActorCritic.step(observations, actions)
    
    # Policy loss
    ratio = torch.exp(predictions['logProbabilities'] - logProbabilitiesOld)
    obj = ratio * advantages
    clippedObj = ratio.clamp(1 - clipRatio, 1 + clipRatio) * advantages
    policyLoss = -1 *  torch.min(clippedObj, obj).mean()
    
    # Entropy loss
    entropyLoss = -entropyCoefficients * predictions['entropy'].mean()

    # Value loss
    valueLoss = (predictions['value'] - returns).pow(2).mean()
    
    # Total loss
    totalLoss = policyLoss + entropyLoss + valueLoss
    
    # Approximated KL for early stopping
    approximatedKL = (logProbabilitiesOld - predictions['logProbabilities']).mean()
    
    clipped = ratio.lt(1 - clipRatio) | ratio.gt(1 + clipRatio)
    clippedFraction = torch.as_tensor(clippedPart, dtype = torch.float32).mean()
    
    info = dict(
        policyLoss = policyLoss.cpu().detach().numpy().item(),
        entropyLoss = entropyLoss.cpu().detach().numpy().item(),
        valueLoss = valueLoss.cpu().detach().numpy().item(),
        totalLoss = totalLoss.cpu().detach().numpy().item(),
        approximatedKL = approxiapproximatedKL.cpu().detach().numpy().item(),
        clippedFraction = clippedFraction.cpu().detach().numpy().item(),
        )
        
    return totalLoss, info


def getBatchGenerator():
    raise NotImplemented

def collectDataBatch():
    raise NotImplemented

def computeMeanDict():
    raise NotImplemented

def trainingUpdate(actorCritic, optimizer, data, miniBatchSize, clipRatio, targetKL, valueCoefficients,   
    entropyCoefficients, gradientClip, meximumNumberOfSteps, device = None):
    infos = {}
    start = time.time()
    numberOfEpochs = 0

    #data = myBuffer.get()
    #policyLossOld, policyInfoOld, valueLossOld = computeLoss(data)
    #policyLossOld = policyLossOld.item()
    #valueLossOld = valueLossOld.item()
        
    # Policy training
    for i in range(meximumNumberOfSteps):
        optimizer.zero_grad()
        batchInfos = []
        batchGenerator = getBatchGenerator(indices = np.arange(len(data['observations'])), batchSize = miniBatchSize)
        for batchIndices in batchGenerator:
            dataBatch = collectDataBatch(data, indices = batchIndices)
            batchTotalLLoss, batchInfo = computeLoss(actorCritic, dataBatch, clipRatio, valueCoefficients, entropyCoefficients, device = device)
            batchTotalLoss.backwards(retain_graph = False)
            batchInfos.append(batchInfo)

        lossInfo = computeMeanDict(batchInfos)
        lossInfo['gradientNorm'] = computeGradientNorm(actorCritic.parameters)


        if lossInfo['approximatedKL'] > 1.5 * targetKL:
            myLog.logPrint('Early stopping at step %d due to reaching maximal KL divergence.' %i)
            break
        
        torch.nn.utils.clip_grad_norm_(actorCritic.parameters(), max_norm = gradientClip)
        optimizer.step()
        optimizer.zero_grad()
        
        numberOfEpochs = numberOfEpochs + 1
        
        # Omer Sella: not implemented: 
        #logging.debug(f'Loss {i}: {loss_info}')
        infos.update(loss_info)

        # Omer Sella: not implemented: 
        if numberOfEpochs > 0:
            logging.info(f'Optimization: policy loss={infos["policy_loss"]:.3f}, vf loss={infos["vf_loss"]:.3f}, '
                     f'entropy loss={infos["entropy_loss"]:.3f}, total loss={infos["total_loss"]:.3f}, '
                     f'num steps={num_epochs}')
        return infos


  
    # Value function training
    #for i in range(valueFunctionTrainIterations):
    #    valueFunctionOptimizer.zero_grad()
    #    _, _, lossValue = computeLoss(data)
    #    lossValue.backwards()
    #    mpi_avg_grads(myActorCritic.valueFunction)
    #    valueFunctionOptimizer.step()
    #        
    # Omer Sella: not implemented: log changes due to update.
    # log.store policyLossOld, valueLossOld, kl, entropy, clipFraction, delta between lossPolicy and policyLossOld, delta between lossValue and valueLossOld
    




def ppo(environmentFunction):
  
    
    # localStepsPerEpochs is the number of steps each MPI process spends in an epoch.
    localStepsPerEpoch = int( numberOfStepsPerEpoch / num_procs() )
    
    # Initialise actor-critic
    myActorCritic = models.actorCritic(int, 2048, int, 16, INTERNAL_ACTION_SPACE_SIZE, 7, [64,64] , 'cpu')
    
    #policyOptimizer = Adam(myActorCritic.policy.parameters(), policyLearningRate)
    #valueFunctionOptimizer = Adam(myActorCritic.value.parameters(), valueFunctionLearningRate)
    
    myLog = utilityFunctions.logger(loggerKeyWords, loggerPath, fileName = 'experiment.txt')
    #myLog.setupPytorchSave(myActorCritic)
    
    myBuffer = ppoBuffer(2048, INTERNAL_ACTION_SPACE_SIZE, localStepsPerEpoch)
        
    startTime = time.time()
    # Omer Sella: dirty fix: cast to float32 because that's what actor-critic model expects.
    observation = environmentFunction.reset().astype(np.float32)
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
            print(xCoordinate)
            print(yCoordinate)
            k = np.int32(k)
            print(coordinates)
            coordinates = np.int32(coordinates)
            newVector[coordinates[0:k]] = 1
            action = np.hstack((np.hstack((xCoordinate, yCoordinate)), newVector))
            nextObservation, reward, done, _ = environmentFunction.step(action)
            nextObservation = nextObservation.astype(np.float32)
            ## book keeping
            episodeReturn = episodeReturn + reward
            episodeLength = episodeLength + 1
            
            ## save to buffer and log
            myBuffer.store(observation = observation, action = action, reward = reward, value = value, logProbability = logProbabilityList, nextObservation = nextObservation)
            
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
                    observation = environmentFunction.reset().astype(np.float32)
                    episodeLength = 0
                    episodeReturn = 0
                    
            myLog.dumpLogger()
        # Save the model accordingly
        if (epoch % SAVE_MODEL_FREQUENCY == 0) or (epoch == (epochs - 1)):
        
            # Omer Sella: not yet implemented: this is a state savingof the experiment. I need to add this ability to the logger.
            #myLog.save_state
            pass    
        
        # ppo policy update
        print("Going into training ...")
        trainingUpdate(actorCritic, optimizer, data, miniBatchSize, clipRatio, targetKL, valueCoefficients, entropyCoefficients, gradientClip, meximumNumberOfSteps, device = None)

if __name__ == '__main__':
    
    # Omer Sella: using lambda didn't give back an environment, I commecnted it out for debug.
    #ppo(lambda : gym.make('gym_ldpc:ldpc-v0'))
    ppo(gym.make('gym_ldpc:ldpc-v0'))      
            
                