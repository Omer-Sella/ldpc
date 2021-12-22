### Environment wrapper
import gym
import concurrent.futures
import numpy as np
import time
from utilityFunctions import numToBits
from numba import cuda
import gym

class environmentVector():
    def __init__(self, environmentGenerationFunction, seeds, cudaDeviceList):
        self.numberOfEnvironments = len(cudaDeviceList)
        self.environmentList = []
        self.environmentIndecies = []
        for i in range(self.numberOfEnvironments):
            self.environmentList.append(environmentGenerationFunction(seeds[i], cudaDeviceList[i]))
            self.environmentIndecies.append(i)
        return

    def singleReset(self, index):
        observation = self.environmentList[index].reset()
        return observation

    def singleStep(self, action, index):
        observedState, reward, done, flags = self.environmentList[index].step(action)
        return [observedState, reward, done, {}]

 

class multiDeviceEnvironment():
    def __init__(self, environmentGenerationFunction, seeds, cudaDeviceList):
        self.environmentVector = environmentVector(environmentGenerationFunction, seeds, cudaDeviceList)
        self.indexList = self.environmentVector.environmentIndecies
        self.cudaDeviceList = cudaDeviceList
        return

    def reset(self):
        with concurrent.futures.ProcessPoolExecutor() as executor:    
            results = executor.map(self.environmentVector.singleReset, self.indexList)
        for r in results:
            print(r)
        
    def step(self, actions):
        start = time.time()
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = {executor.submit(self.environmentVector.singleStep, actions[deviceNumber], deviceNumber): deviceNumber for deviceNumber in self.cudaDeviceList}
            for result in concurrent.futures.as_completed(results):
                print(result)
        #with concurrent.futures.ProcessPoolExecutor() as executor:    
        #    stepResults = executor.map(self.environmentVector.singleStep, actions, self.indexList)
        end = time.time()   
        #for r in stepResults:
        #    print(r)
        print("*** Time it took concurrently: " +str(end-start))
        return results



def testEnvironmentVector():
    # Init an environmentVector containing as many environments as there are cuda devices
    # Check some parameters and functionality
    assert (len(cuda.gpus) > 0 ), "Omer Sella: either importing cuda from numba didn't work, or there are 0 cuda devices detectable."
    environmentGenerationFunction = lambda x = 8200, y = 0: gym.make('gym_ldpc:ldpc-v0', seed = x, gpuDevice = y)
    seeds = [61017406, 7134066, 90210, 42]
    # Make sure we have enough seeds for the insane case where there are more than 4 devices on a machine
    if len(cuda.gpus) > len(seeds):
        seeds = list(range(len(cuda.gpus)))
    envVecor = environmentVector(environmentGenerationFunction, seeds, list(range(len(cuda.gpus))))
    for i in list(range(len(cuda.gpus))):
        print("Seed for the " + str(i) + "th environment is : ")
        print(envVecor.environmentList[i].seed)
    for i in list(range(len(cuda.gpus))):
        print("Attempting reset the " + str(i) + "th environment: ")
        print(envVecor.singleReset(i))
    
    NUMBER_OF_HOT_BITS = 7
    localRandom = np.random.RandomState(0)
    [i] = localRandom.choice(2, 1)
    [j] = localRandom.choice(16, 1)
    vector = np.zeros(511, dtype = int)
    xCoordinate = numToBits(i, 1)
    yCoordinate = numToBits(j, 4)
    hotBits = localRandom.choice(511, NUMBER_OF_HOT_BITS, replace = False)
    vector[hotBits] = 1
    action = np.hstack((np.hstack((xCoordinate, yCoordinate)), vector))
    for i in list(range(len(cuda.gpus))):
        print("Attempting step for the " + str(i) + "th environment: ")
        print(envVecor.singleStep(action, i))
    return
    

def testMultiDeviceEnvironment():
    #from utilityFunctions import numToBits
    #from numba import cuda
    #assert (len(cuda.gpus) > 0 ), "Omer Sella: either importing cuda from numba didn't work, or there are 0 cuda devices detectable."
    environmentGenerationFunction = lambda x = 8200, y = 0: gym.make('gym_ldpc:ldpc-v0', seed = x, gpuDevice = y)
    numberOfCudaDevices = 4
    seeds = [61017406, 7134066, 90210, 42]
    # Make sure we have enough seeds for the insane case where there are more than 4 devices on a machine
    #if len(cuda.gpus) > len(seeds):
    #    seeds = list(range(len(cuda.gpus)))
    multiDevEnv = multiDeviceEnvironment(environmentGenerationFunction, seeds, list(range(numberOfCudaDevices)))

    #print("*** testing multi device environment reset function")
    #multiDevEnv.reset()
    
    #### Preparing an action
    NUMBER_OF_HOT_BITS = 7
    localRandom = np.random.RandomState(0)
    [i] = localRandom.choice(2, 1)
    [j] = localRandom.choice(16, 1)
    vector = np.zeros(511, dtype = int)
    xCoordinate = numToBits(i, 1)
    yCoordinate = numToBits(j, 4)
    hotBits = localRandom.choice(511, NUMBER_OF_HOT_BITS, replace = False)
    vector[hotBits] = 1
    action = np.hstack((np.hstack((xCoordinate, yCoordinate)), vector))
    # Now we need to make it into a list of actions:
    actions = [action] * len(cuda.gpus)
    print("*** testing multi device environment step function")
    #print("*** actions are:")
    #print(actions)
    results = multiDevEnv.step(actions)
    for r in results:
        print(r)
    #print(results)

    
if __name__ == '__main__':
    #print("*** Testing environmentVector...")
    #testEnvironmentVector()
    print("*** Testing multi device environment...")
    testMultiDeviceEnvironment()
