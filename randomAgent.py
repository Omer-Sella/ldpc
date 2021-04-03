import numpy as np
import torch
import gym
import time
#from spinup.utils.logx import EpochLogger

import fileHandler
import os


# Omer Sella 06/01/2021
# Random agent.

projectDir = os.environ.get('LDPC')
print(projectDir)
if projectDir == None:
    import pathlib
    projectDir = pathlib.Path(__file__).parent.absolute()
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, projectDir)
import utilityFunctions


def numToBits(number, numberOfBits):
    assert number < 16
    assert number >= 0
    newNumber = np.zeros(numberOfBits, dtype = int)
    for j in range(numberOfBits - 1, -1, -1):
        newNumber[j] = newNumber[j] + (number % 2)
        number = number >> 1
    return newNumber
    

def randomAgent(environmentFunction, seed = 90210, maximumStepsPerEpoch = 4000, numberOfEpochs = 50, saveFrequency = 10, maximumTrajectoryLenth = 1000, savePath = 'D:/swift/evaluations/oneSparseCirculantAway/'):
    """
    Random agent.
    Arguments:
        environmentFunction : A function which creates a copy of the environment
        
        seed (int) : Seed for random number generation
        
        maximumStepsPerEpoch : maximal number of guesses (attempts) the agent is permitted in an Epoch. # note this is not
        a rollout AKA epoch AKA trajectory.
        
        numberOfRollouts : for a random agent this is the same as increasing the maximal steps per rollout, but for logging purposes we keep this structure.
        
        saveFrequency : currently has no importance here.
    """
    keys = ['Epoch', 'Episode return', 'Episode Length', 'VVals', 
            'Total Environemnet interactions', 'LossPi', 'LossV', 
            'DeltaLossPi', 'SeltaLossV', 'Entropy', 'KL', 'ClipFrac',
            'StopIter', 'Time']
    simpleKeys = ['Observation', 'iAction', 'jAction', 'hotBitsAction', 'Reward']
    logger = utilityFunctions.logger(simpleKeys, logPath = "C:/User/optimus/testLogger/")
    #logger.save_config(locals())
    
    # Create a local np.random. Do not change the numpy global random state !
    localRandom = np.random.RandomState(seed)
    
    # Instatiate an environment
    env = environmentFunction()
    
    observation = env.reset()
    
    trajectoryReturn = 0
    trajectoryLength = 0
    timeout = False
    for epoch in range(numberOfEpochs):
        for t in range(maximumStepsPerEpoch):
            # Get next action from the actor
            for numOfHotBits in [7]: #[3,4,5,6,7]:
                experimentPath = savePath + str(numOfHotBits) + '_hot/'
                for i in range(2):
                    for j in range(15):
                        logger.keyValue('Observation', observation)
                        vector = np.zeros(511, dtype = int)
                        xCoordinate = numToBits(i, 1)
                        yCoordinate = numToBits(j, 4)
                        hotBits = localRandom.choice(511, numOfHotBits, replace = False)
                        logger.keyValue('iAction', i)
                        logger.keyValue('jAction', j)
                        logger.keyValue('hotBitsAction', hotBits)
                        vector[hotBits] = 1
                        print("*** i is :")
                        print(i)
                        print("*** j is :")
                        print(j)
                        print("*** hotbits are :")
                        print(hotBits)
                        print(vector)
                        print(len(vector))
                        #assert (len(vector) == 511), "Vector length is" + str(len(vector))
                        action, value, logp = np.hstack((np.hstack((xCoordinate, yCoordinate)), vector)) , 0, 0#actor.step()
                        startTime = time.time()
                        nextObservation, nextReturn, nextDone, _ = env.step(action)
                        
                        logger.keyValue('Reward', nextReturn)
                        logger.dumpLogger()
                        uncompressedFirstRow, uncompressedSecondRow = env.uncompress()
                        if (np.all(uncompressedFirstRow == env.state[0, :])):
                            pass
                        else:
                            input("Press Enter to continue...")
                        if (np.all(uncompressedSecondRow == env.state[511, :])):
                            pass
                        else:
                            input("Press Enter to continue...")
                        endTime = time.time()
                        evaluationTime = endTime - startTime
#                        fileName = fileHandler.saveCodeInstance(env.state, env.circulantSize, env.codewordSize, env.berStats, experimentPath, evaluationTime, 0)
                        trajectoryReturn = trajectoryReturn + nextReturn
                        trajectoryLength = trajectoryLength + 1
        
                        timeout = (trajectoryLength == maximumTrajectoryLenth)
                        epochEnd = (t == (maximumStepsPerEpoch - 1))
                
                        if True: # Replaces timeout or epochEnd or nextDone:
                            observation = env.reset()
                            trajectoryReturn = 0
                            trajectoryLength = 0
                        
                        observation = nextObservation
                
            
                
            
if __name__ == '__main__':
    
    randomAgent(lambda : gym.make('gym_ldpc:ldpc-v0'))