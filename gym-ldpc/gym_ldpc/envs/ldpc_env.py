# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 12:24:04 2019

@author: Omer
"""

import gym
#from gym import error, spaces, utils
#from gym.utils import seeding
from scipy.linalg import circulant
from scipy import integrate


import numpy as np
import copy

GENERAL_LDPC_ENV_TYPE = np.float32
LDPC_ENV_INT_DATA_TYPE = np.int32
LDPC_ENV_SEED_DATA_TYPE = np.int32
LDPC_ENV_NUMBER_OF_ITERATIONS = 50LDPC_ENV_NUMBER_OF_TRANSMISSIONS = 40
# How many seconds is a batch. Should be tested with number of iterations as well.
#This is a way to limit the entire training process on cluster use 
LDPC_ENV_MAXIMUM_ACCUMULATED_DECODING_TIME = 64 * LDPC_ENV_NUMBER_OF_TRANSMISSIONS
# Omer Sella: seeds are required by concurrent futures to be between 0 and 2**32 - 1 
LDPC_ENV_MAX_SEED = 2**31 - 1
#seed = 7134066

LDPC_POLYNOMIAL_ORDER = 1

import os
projectDir = os.environ.get('LDPC')
if projectDir == None:
    import pathlib
    projectDir = pathlib.Path(__file__).parent.absolute()
## Omer Sella: added on 01/12/2020, need to make sure this doesn't break anything.
import sys
sys.path.insert(1, projectDir)
## Omer Sella: end added.

import fileHandler
import ldpcCUDA
import time
from binarySpace import binarySpace
from uint8Space import uint8Space
import common

class LdpcEnv(gym.Env):
    """
        Description: an LDPC code is represented by a binary matrix. Loosley speaking, a code is good if its output BER is low at low SNR.
        
        Observation space:
            
            a 2X16 array of compressed 511 bits.
            I.e.: each 511 bits first row of a circulant is appended a 0 (on the left),
            and then packed into bytes using np.packbits()
            
        Actions:
            Type: Discrete( {0,1} X {0,..,15} X 2^511)
            xCoordinate, yCoordinate, first row of new circulant to be installed at coordinates x,y
        
        Reward:
            The reward is the slope of the BER-SNR graph.
            
        Starting state:
            Binary matrix corresponding to near-earth.
            
        Episode termination:
            Either: 1. BER at first SNR point is higher than TBD1.
                    2. Episode length is greater than TBD2.
                    3. Good code found (defined by target slope, TBD3).
    
    """
    
  ## Omer Sella: probably change to binary array instead of rgb array.
    metadata = {'render.modes': ['rgb']}

  #def __init__(self, SNRpointsOfInterest = [5, 10, 15], H, xLimit, yLimit, circulantSize):
    def __init__(self, replacementOnly=False, seed=7134066, numberOfCudaDevices = 4, resetHammingDistance = 'MAXIMUM'):
        
        H = fileHandler.readMatrixFromFile(projectDir + '/codeMatrices/nearEarthParity.txt', 1022, 8176, 511, True, False, False)
        self.replacementOnly = replacementOnly
        self.messageSize = 7156
        self.codewordSize = 8176
        #self.SNRpoints = np.array([3.0, 3.2, 3.4, 3.6, 3.8], dtype = GENERAL_LDPC_ENV_TYPE)
        self.SNRpoints = np.array([3.0, 3.2, 3.4], dtype = GENERAL_LDPC_ENV_TYPE) #Omer Sella: removed last two snr points
        self.BERpoints = np.ones(len(self.SNRpoints), dtype = GENERAL_LDPC_ENV_TYPE)
        self.state = copy.deepcopy(H)
        self.resetValue = copy.deepcopy(H)
        self.circulantSize = 511
        # Omer Sella: should be made clear from the action space details
        self.xBits = 1
        # Omer Sella: should be made clear from the action space details
        self.yBits = 4
        # Omer Sella: should be made clear from the action space details
            
        if replacementOnly == False:

            self.action_space = binarySpace(self.xBits + self.yBits + self.circulantSize)
            self.actionBits = self.xBits + self.yBits + self.circulantSize
        else:
            self.action_space = binarySpace(self.xBits + self.xBits + self.yBits + self.yBits)
            self.actionBits = 2 * self.xBits + 2 * self.yBits
            
        
        self.observation_space = uint8Space(2048)
        self.observationUint8 = 2048
        self.paddingLocations = (np.arange(16) + 1) * (self.circulantSize +1 ) - 1
        self.compressionMask = np.ones(8192, dtype = bool)
        self.compressionMask[self.paddingLocations] = False
        self.observed_state = self.compress()
        self.ldpcDecoderNumOfIterations = LDPC_ENV_NUMBER_OF_ITERATIONS
        self.ldpcDecoderNumOfTransmissions = LDPC_ENV_NUMBER_OF_TRANSMISSIONS
        self.xCoordinateBinaryToInt = np.flipud(2**np.arange(self.xBits))
        self.yCoordinateBinaryToInt = np.flipud(2**np.arange(self.yBits))
        self.seed = seed
        self.viewer = None
        self.steps_beyond_done = None
        self.rewardForIllegalAction = -2.0
        self.rewardForBadCandidate = -2.0
        self.method = 'Incremental'
        self.refernce = 0.025
        #self.fig, self.ax = common.spawnGraphics(self.state, self.circulantSize, self.circulantSize)
        self.scatterSnr = self.SNRpoints
        self.scatterBer = self.BERpoints
        self.scatterItr = np.ones(len(self.SNRpoints), dtype = GENERAL_LDPC_ENV_TYPE)
        self.accumulatedEvaluationTime = 0
        #berStats = self.evaluateCode()
        #self.scatterSnr, self.scatterBer, self.scatterItr, snrAxis, averageSnrAxis, berData, averageNumberOfIterations = berStats.getStatsV2()
        #self.calcReward("red")
        self.maximumNNZ = 64000
        #nearEart density 0.0005589714924538849
        self.targetDensity = 6.0/511 #0.0005589714924538849
        self.berStats = common.berStatistics()
        self.pConst = np.poly1d([1])
        #unpaddedFirstRow, unpaddedSecondRow = self.uncompress()
        #assert np.all(unpaddedFirstRow == self.state[0,:])
        #assert np.all(unpaddedSecondRow == self.state[511,:])
        self.localPRNG = np.random.RandomState(seed)
        self.hotBitsRange = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        self.cudaDevices = numberOfCudaDevices
        self.counter = 0
        self.resetHammingDistance = resetHammingDistance

        
        
        return
    ## 01/01/2021 Omer Sella: I changed the step function to just check xEntropy of the next action, this is to see if the agent is learning sparsity.
    ## Happy new year !
    ## 04/05/2021 Omer Sella: I commented the xEntropy step function, back to evaluating actual codes.
    """
    def step(self, action):
        done = False
        # Get first line of new circulant
        circulantFirstRow = action[self.xBits + self.yBits : ]
        circulantDensity = np.sum(circulantFirstRow) / self.circulantSize
        # Calculate the negative of cross entropy bentween bernouli with self.targetDensity and the circulant we got as a specific step !
        logTargetProbs = np.log2(np.where(circulantFirstRow == 1, self.targetDensity, 1 - self.targetDensity))
        sampledProbs = np.where(circulantFirstRow == 1, circulantDensity, 1 - circulantDensity)
        reward = np.sum(sampledProbs * logTargetProbs)
        #Omer Sella: note that cross entropy is: -1 * np.sum(sampledProbs * logTargetProbs)
        print(reward)
        if circulantDensity > 2 * self.targetDensity:
            done = True
        return self.observed_state, reward, done, {}
        
    """
    def step(self, action):
        done = False
        ## Omer Sella: the assertion that the action is of type int vector is for safety during developement and should be removed upon release
        #print("*** ")
        #print(action.dtype)
        #assert(action.dtype == 'int32'), "Action vector was not int32 type. Any int type will do."
        assert(action.shape[0] == self.xBits + self.yBits + self.circulantSize)
        # First we need to unpack the action from a vector into x,y coordinates, and then either a circulant or replacement coordinates (depending on the environment type as set in self.replacementOnly)
        xCoordinateBinary = action[0 : self.xBits]
        #print(xCoordinateBinary.shape)
        xCoordinate = self.xCoordinateBinaryToInt.dot(xCoordinateBinary)
        yCoordinateBinary = action[self.xBits : self.xBits + self.yBits]
        #print(yCoordinateBinary.shape)
        yCoordinate = self.yCoordinateBinaryToInt.dot(yCoordinateBinary)
        # 26/05/2022 disabled density calc - it is no longer used.
        #density = np.sum(self.state) / (self.messageSize * self.codewordSize)
        #print("*** density of state before action :" + str(density))
        if self.replacementOnly == True:
            xrCoordinateBinary = action[self.xBits + self.yBits : self.xBits + self.yBits + self.xBits]
            xrCoordinate = self.xCoordinateBinaryToInt.dot(xrCoordinateBinary)
            yrCoordinateBinary = action[self.xBits + self.yBits + self.xBits : + self.xBits + self.yBits + self.xBits + self.yBits]
            yrCoordinate = self.yCoordinateBinaryToInt.dot(yrCoordinateBinary)
            
            xyCirculant = self.state[xCoordinate * self.circulantSize : (xCoordinate + 1) * self.circulantSize, yCoordinate * self.circulantSize : (yCoordinate + 1) * self.circulantSize]
            xryrCirculant = self.state[xrCoordinate * self.circulantSize : (xrCoordinate + 1) * self.circulantSize, xrCoordinate * self.circulantSize : (xrCoordinate + 1) * self.circulantSize]
            #print("Replace circulant " + str(xCoordinate) + " , " + str(yCoordinate) + " with " +str(xrCoordinate) + " , " + str(yrCoordinate))
            status = self.replaceCirculant(xCoordinate, yCoordinate, xryrCirculant)            
            if status == 'OK':
                status = self.replaceCirculant(xrCoordinate, yrCoordinate, xyCirculant)
            
            else:
                status = 'FAILED'
            #print(status)
            
        else:
            #print(self.xBits)
            #print(self.yBits)
            circulantFirstRow = action[self.xBits + self.yBits : ]
            #print(circulantFirstRow.shape)
            #assert(len(circulantFirstRow)) == self.circulantSize
            newCirculant = circulant(circulantFirstRow).T
            status = self.replaceCirculant(xCoordinate, yCoordinate, newCirculant)
        
        # Now evaluate the code
        
        if status == 'OK':
            # Omer Sella: circulant change was legal and successful. Evaluate new code.
            self.berStats = self.evaluateCode()
            self.scatterSnr, self.scatterBer, self.scatterItr, snrAxis, averageSnrAxis, berData, averageNumberOfIterations = self.berStats.getStatsV2()
            #test = berStats.stats
            #print(test)
            self.BERpoints = berData
            # berData needs to be at least 2 points for a valid reward. berData is: " + str(self.BERpoints))
            # and its length is " + str(len(self.BERpoints)))
            reward = self.calcReward()
            #density = np.sum(self.state) / (self.messageSize * self.codewordSize)
                
        else:
            reward = self.rewardForIllegalAction
        
        #print("*** Finishing step. ")
        #print("*** *** *** *** ***Reward == " + str(reward))
        #print("*** *** *** *** *** Done == " + str(done))
        #print("*** accumulated decoding time == " + str(self.accumulatedEvaluationTime))
        #Omer Sella: disabled accumulated decoding time cap.
        #if self.accumulatedEvaluationTime > LDPC_ENV_MAXIMUM_ACCUMULATED_DECODING_TIME:
        #    done = True
        #    #print("**** DECODING TIME EXHAUSTED" + str(self.accumulatedEvaluationTime))
        self.observed_state = self.compress()
        return self.observed_state, reward, done, {}
    
      
    def reset(self):
        self.state = copy.deepcopy(self.resetValue)
        self.observed_state = self.compress()
        self.BERpoints = np.ones(len(self.SNRpoints), dtype = GENERAL_LDPC_ENV_TYPE)
        self.accumulatedEvaluationTime = 0
        return self.observed_state

    def render(self, mode='rgb', close=False):
      #common.plotSNRvsBER(self.SNRpoints, self.BERpoints, inputLabel = 'Current State', fileName = "snrVSber.png")  
      return #timeStamp
  
    

    
    def replaceCirculant(self, xCoordinate, yCoordinate, newCirculant):
        
        m,n = self.resetValue.shape
        #print('*****')
        #print(newCirculant.shape)
        circulantSize =  newCirculant.shape[0]
        status = 'NA'
        #print("*** circulant size is: ")
        #print(circulantSize)
        #print("*** xCoordinate is: ")
        #print(xCoordinate)
        #print("*** yCoordinate is: ")
        #print(yCoordinate)
        if (xCoordinate > (2 ** self.xBits)) or (yCoordinate > (2 ** self.yBits)):
            status = 'Illegal action'
        else:      
            #print(xCoordinate * circulantSize)
            #print((xCoordinate + 1) * circulantSize)
            #print(yCoordinate * circulantSize)
            #print((yCoordinate + 1) * circulantSize)
            self.state[xCoordinate * circulantSize : (xCoordinate + 1) * circulantSize, yCoordinate * circulantSize : (yCoordinate + 1) * circulantSize] = newCirculant
            #common.updateCirculantImage(self.ax, self.fig, xCoordinate, yCoordinate, newCirculant)
            status = 'OK'
        
        return status
    
    def calcReward(self, colour = None):
        # Omer Sella: fit a line to the snr / ber data, and return the slope.
        if len(self.BERpoints) < 2:
            # You need at least two points to fit a line
            reward = self.rewardForBadCandidate
        else:
                        
            # Fit a line through the data #OSS 26/12/2021 this is now proivided by a function from common
            #p = np.polyfit(self.scatterSnr, self.scatterBer, LDPC_POLYNOMIAL_ORDER)
            # OSS 26/12/2021 adjusted the reward function to be less sensitive to 0 BER data points
            snr, ber, p1, trendP, itr = common.recursiveLinearFit(self.scatterSnr, self.scatterBer)


            # Omer Sella: 16/06/2021 decided to use np polynomials. Also changed the reward to the area between
            # the constant 1 and the fitted line.
            #slope = p[0]
            #bias = p[1]
            # OSS: p1 is now an output of a function from common.py
            #p1 = np.poly1d(p)
            pTotalInteg = (self.pConst - p1).integ()
            reward = pTotalInteg(self.SNRpoints[-1]) - pTotalInteg(self.SNRpoints[0])
            #reward =  0.5 * slope * (self.SNRpoints[-1] ** 2)  + bias * self.SNRpoints[-1] - ( 0.5 * slope * (self.SNRpoints[0] ** 2)  + bias * self.SNRpoints[0])
            #reward = -1 * reward
            
        #common.updateBerVSnr(self.ax, self.fig, 1, 16, self.scatterSnr, self.scatterBer, colour)
        #common.updateReward(self.ax, self.fig, 0, 16, self.SNRpoints, slope * self.SNRpoints + bias, colour)
        return reward
    
    
    def evaluateCodeSingleTransmissionUsingSeed(self, seed):
        return ldpc.testCodeUsingMultiprocessing(seed, np.flipud(self.SNRpoints), self.messageSize, self.codewordSize, self.ldpcDecoderNumOfIterations, self.ldpcDecoderNumOfTransmissions, self.state, self.method, self.refernce, G = 'None')
    
    

    def evaluateCode(self):
        seeds = self.localPRNG.randint(0, LDPC_ENV_MAX_SEED, self.cudaDevices, dtype = LDPC_ENV_SEED_DATA_TYPE)
        seed = self.localPRNG.randint(0, LDPC_ENV_MAX_SEED, 1, dtype = LDPC_ENV_SEED_DATA_TYPE)
        #seed = np.random.randint(0, LDPC_ENV_MAX_SEED, 1, dtype = LDPC_ENV_SEED_DATA_TYPE)
        #transmissions = np.arange(0, self.ldpcDecoderNumOfTransmissions, 1, dtype = LDPC_ENV_INT_DATA_TYPE)
        
        #SNR_iterable = list([self.SNRpoints])  * self.ldpcDecoderNumOfTransmissions
        #messageSize_iterable = list([self.messageSize])  * self.ldpcDecoderNumOfTransmissions
        #codewordSize_iterable = list([self.codewordSize]) * self.ldpcDecoderNumOfTransmissions
        #numberOfIterations_iterable = list([self.ldpcDecoderNumOfIterations]) * self.ldpcDecoderNumOfTransmissions
        #H_iterable = list([self.state]) * self.ldpcDecoderNumOfTransmissions
        
        
        start = time.time()        
        berStats = common.berStatistics(self.codewordSize)
        # OSS: I'm commenting out evaluateCodeCuda in order to use the wrapper that utilises multiple GPUs
        #berStats = ldpcCUDA.evaluateCodeCuda(seed, self.SNRpoints, self.ldpcDecoderNumOfIterations, self.state, self.ldpcDecoderNumOfTransmissions, G = 'None', cudaDeviceNumber = self.cudaDevices)
        berStats = ldpcCUDA.evaluateCodeCudaWrapper(seeds, self.SNRpoints, self.ldpcDecoderNumOfIterations, self.state, self.ldpcDecoderNumOfTransmissions, G = 'None' , numberOfCudaDevices = self.cudaDevices)
        snrAxis, averageSnrAxis, berData, averageNumberOfIterations = berStats.getStats()
        #end = time.time()
        #print('Time it took for code evaluation == %d' % (end-start))
        #print("berDecoded " + str(berData))
        #self.accumulatedEvaluationTime = self.accumulatedEvaluationTime + (end-start)

        return berStats
    
    def compress(self):
        firstRow = self.state[0,:]
        secondRow = self.state[511,:]
        
        paddedFirstRow = np.zeros(8192, dtype = np.dtype(firstRow[0])) #np.insert(firstRow, self.paddingLocations, 0)
        paddedFirstRow[self.compressionMask] = firstRow
        
        paddedSecondRow = np.zeros(8192, dtype = np.dtype(secondRow[0])) #np.insert(secondRow, self.paddingLocations, 0)
        paddedSecondRow[self.compressionMask] = secondRow
        
        firstRow = np.packbits(paddedFirstRow)
        secondRow = np.packbits(paddedSecondRow)
        ##Omer Sella: Changed vstack to hstack, since ppo wants the data flattened anyway.
        #self.observed_state = np.vstack((firstRow, secondRow))
        self.observed_state = np.hstack((firstRow, secondRow))
        return self.observed_state
    
    def uncompress(self):
        firstRow = np.unpackbits(self.observed_state[0 : len(self.observed_state) // 2 ])
        secondRow = np.unpackbits(self.observed_state[len(self.observed_state) // 2 :])
        unpaddedFirstRow = firstRow[self.compressionMask]
        unpaddedSecondRow = secondRow[self.compressionMask]
        return unpaddedFirstRow, unpaddedSecondRow
        
    
    def constantFunction(self, alpha):
        def g(x):
            return alpha
        return g
    
    
def testCompressionRoundrip():
    testEnvironment = LdpcEnv()
    for i in range(1000):
        firstRow, secondRow = testEnvironment.uncompress()
        assert np.all(firstRow == testEnvironment.state[0,:]), "First row does not match"
        assert np.all(secondRow == testEnvironment.state[511,:]), "Second row does not match"
        testEnvironment.state = np.random.randint(0, 2, [1022,8176])
        testEnvironment.observed_state = testEnvironment.compress()
    return 'OK'
        
    
    
    
    
    
    
        
