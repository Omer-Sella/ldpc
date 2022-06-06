import numpy as np
import torch
import gym
import time
#from spinup.utils.logx import EpochLogger

import fileHandler
import os
import multiprocessing

# Omer Sella 06/01/2021
# Random agent.

projectDir = os.environ.get('LDPC')
#print(projectDir)
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
    

def randomAgent(environmentFunction, seed = 90210, maximumStepsPerEpoch = 5, numberOfEpochs = 50, saveFrequency = 10, maximumTrajectoryLenth = 1000, savePath = './randomAgent/', envCudaDevices = 1):
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
    logger = utilityFunctions.logger(simpleKeys, logPath = (savePath + '/log/'))
    #logger.save_config(locals())
    myPlotter = utilityFunctions.plotter(numberOfEpochs)
    # Create a local np.random. Do not change the numpy global random state !
    localRandom = np.random.RandomState(seed)
    
    # Instatiate an environment
    env = environmentFunction(x = seed, y = envCudaDevices)
    
    
    hotBitsVector = np.array([3,4,5,6,7])
    
    timeout = False
    for epoch in range(numberOfEpochs):
        observation = env.reset()
        trajectoryReturn = 0
        trajectoryLength = 0
        for t in range(maximumStepsPerEpoch):
            [i] = localRandom.choice(2, 1)
            [j] = localRandom.choice(16, 1)
            [numOfHotBits] = localRandom.choice(hotBitsVector, 1)
            hotBits = localRandom.choice(511, numOfHotBits, replace = False)
            # Get next action from the actor
            #for numOfHotBits in [3,4,5]:#,6,7]:
            experimentPath = savePath + str(numOfHotBits) + '_hot/'
            #    for i in range(2):
            #        for j in range(15):
            logger.keyValue('Observation', observation)
            vector = np.zeros(511, dtype = int)
            xCoordinate = numToBits(i, 1)
            yCoordinate = numToBits(j, 4)
            hotBits = localRandom.choice(511, numOfHotBits, replace = False)
            logger.keyValue('iAction', i)
            logger.keyValue('jAction', j)
            logger.keyValue('hotBitsAction', hotBits)
            vector[hotBits] = 1
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
    
            #if True: # Replaces timeout or epochEnd or nextDone:
            #    observation = env.reset()
            #    trajectoryReturn = 0
            #    trajectoryLength = 0
            
            observation = nextObservation

             # Log info about epoch
            
                
            
if __name__ == '__main__':
    import argparse
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default= 'gym_ldpc:ldpc-v0')
    parser.add_argument('--seed', '-s', type=int, default=30)
    parser.add_argument('--cpu', type=int, default=1) #Omer Sella: was 4 instead of 1
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--envCudaDevices', type=int, default=1)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()
    import os
    experimentTime = time.time()
    PROJECT_PATH = os.environ.get('LDPC')
    experimentDataDir = PROJECT_PATH + "/randomAgent/experiments/%i" %int(experimentTime)
    
    randomAgent(lambda x = 8200, y = 0: gym.make(args.env, seed = x, numberOfCudaDevices = y),
        seed=args.seed, maximumStepsPerEpoch=args.steps, numberOfEpochs=args.epochs,
        envCudaDevices = args.envCudaDevices, savePath = experimentDataDir)