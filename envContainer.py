### Environment wrapper
import gym
import concurrent.futures

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
        observedState, reward, done, flags = self.environmentList[index].step(asction)
    return observedState, reward, done, {}

 

class multiDeviceEnvironment():
    def __init__(self, environmentGenerationFunction, seeds, cudaDeviceList):
        self.environmentVector = environmentVector(environmentGenerationFunction, seeds, cudaDeviceList)
        self.indexList = self.environmentVector.environmentIndecies
    return

    def reset(self):
        with concurrent.futures.ProcessPoolExecutor() as executor:    
            results = executor.map(self.environmentVector.reset, self.indexList)
        for r in results:
            print(r)
        
    def step(self, actions):
        with concurrent.futures.ProcessPoolExecutor() as executor:    
            results = executor.map(self.environmentVector.step, actions, self.indexList)
        for r in results:
            print(r)


def testEnvironmentVector():
    # Init an environmentVector containing as many environments as there are cuda devices
    # Check some parameters and functionality
    from numba import cuda
    assert (len(cuda.gpus) > 0 ), "Omer Sella: either importing cuda from numba didn't work, or there are 0 cuda devices detectable."
    environmentGenerationFunction = lambda x = 8200, y = 0: gym.make(args.env, seed = x, gpuDevice = y)
    seeds = [61017406, 7134066, 90210, 42]
    # Make sure we have enough seeds for the insane case where there are more than 4 devices on a machine
    if len(cuda.gpus) > len(seeds):
        seeds = list(range(len(cuda.gpus)))
    envVecor = environmentVector(environmentGenerationFunction, seeds, list(range(len(cuda.gpus))))
    for i in list(range(len(cuda.gpus))):
        print("Seed for the " + str(i) + "th environment is : ")
        print(environmentContainer.environmentList[i].seed)
    for i in list(range(len(cuda.gpus))):
        print("Attempting reset the " + str(i) + "th environment: ")
        print(environmentContainer.singleReset(i))
    
    localRandom = np.random.RandomState(0)
    [i] = localRandom.choice(2, 1)
    [j] = localRandom.choice(16, 1)
    [numOfHotBits] = localRandom.choice(hotBitsVector, 1)
    hotBits = localRandom.choice(511, numOfHotBits, replace = False)
    vector = np.zeros(511, dtype = int)
    xCoordinate = numToBits(i, 1)
    yCoordinate = numToBits(j, 4)
    hotBits = localRandom.choice(511, numOfHotBits, replace = False)
    vector[hotBits] = 1
    action = np.hstack((np.hstack((xCoordinate, yCoordinate)), vector))
    for i in list(range(len(cuda.gpus))):
        print("Attempting step for the " + str(i) + "th environment: ")
        print(environmentContainer.singleStep(action, i))
    return
    

    
if __name__ == '__main__':
    print("*** Testing environmentVector...")
    testEnvironmentVector()
