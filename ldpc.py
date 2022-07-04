## oss 08/07/2019
import numpy as np
import time
import concurrent.futures
import os
import copy
from numba import jit, int32, float32, jitclass, types, typed, boolean, float64, int64
#import math
import wifiMatrices

projectDir = os.environ.get('LDPC')
if projectDir == None:
    import pathlib
    projectDir = pathlib.Path(__file__).parent.absolute()
projectDirEvals = str(projectDir) + "evaluations/"

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, projectDir)
#import io

import fileHandler
import common
LDPC_GLOBAL_PRNG = np.random.RandomState(7134066)
# LDPC_**_DATA_TYPE stores the data type over which all arithmetic is done.
# It is a nice way of changing the data type of the entire implementation at one place.
LDPC_DATA_TYPE = np.int64
LDPC_INT_DATA_TYPE = np.int64
LDPC_DECIMAL_DATA_TYPE = np.float64
LDPC_SEED_DATA_TYPE = np.int64
# Omer Sella: Major breakdown warning: the bool data type is used to create a mask. Replacing it with int32 breaks the decoder.
LDPC_BOOL_DATA_TYPE = boolean
# Omer Sella: seeds can be integers between 0 and 2**31 - 1
LDPC_MAX_SEED = 2**31 - 1

NUMBA_INT = int64
NUMBA_FLOAT = float64
NUMBA_BOOL = boolean




@jit(nopython = True)
def slicer(vector, length):
    ## Omer Sella: slicer puts a threshold, everything above 0 is translated to 1,  otherwise 0 (including equality). Do not confuse with the reserved function name slice !
    slicedVector = np.ones(length, dtype = LDPC_INT_DATA_TYPE)#LDPC_DECIMAL_DATA_TYPE)
    slicedVector[np.where(vector <= 0)] = 0
    return slicedVector

#@jit()
def addAWGN(vector, length, SNRdb, prng):
    ## The input SNR is in db so first convert:
    SNR = 10 ** (SNRdb/10)
    ## Now use the definition: SNR = signal^2 / sigma^2
    sigma = np.sqrt(0.5 / SNR)
    #print(sigma)
    noise = prng.normal(0, sigma, length)
    sigmaActual = np.sqrt((np.sum(noise ** 2)) / length)
    noisyVector = vector + noise
    return noisyVector, sigma, sigmaActual

@jit(nopython = True)
def modulate(vector, length):
    modulatedVector = np.ones(length, dtype = LDPC_DECIMAL_DATA_TYPE)
    modulatedVector[np.where(vector == 0)] = -1
    return modulatedVector


specForVariableNode = [
    ('identity', NUMBA_INT),
    ('fromChannel', float64),
    ('presentState', float64),
    
]
@jitclass(specForVariableNode)
class variableNode:
# a variableNode is merely a memory element. 
    def __init__(self, identity):
        self.identity = identity
       
        self.fromChannel = 0
        self.presentState = 0
        return
        
    def update(self, value):
        #assert(np.dtype(value) == LDPC_DECIMAL_DATA_TYPE)
        self.presentState = self.presentState + value
        return
    

## 
# key and value types for variableIDToIndex: 
key_value_types = (NUMBA_INT, NUMBA_INT)
specForCheckNode = [
    ('identity', NUMBA_FLOAT),
    ('numberOfConnectedVariableNodes', NUMBA_INT),
    ('incomingValues', NUMBA_FLOAT[:,:]),
    ('outgoingValues', NUMBA_FLOAT[:,:]),
    ('signVector', NUMBA_FLOAT[:]),
    ('magnitudeVector', NUMBA_FLOAT[:]),
    ('connectedVariableNodes', NUMBA_INT[:]),
    ('variableIDtoIndexDictionary', types.DictType(*key_value_types)),
    ('mask', LDPC_BOOL_DATA_TYPE[:]),
    ]
@jitclass(specForCheckNode)
class checkNode:
# A check node is where most of the logic is done.
# Every check node has an id (this is redundant in serial execution, but may make things easier when moving to an asynchronous implementation)
# A checkNode stores the values of the variableNodes in the incoming values with the sender identities.
# When all incoming values are received, it may calculate the sign vector of these values as well as the magnitude.
# Each outgoing message is a two entry array (a line in outGoingMessages), that contains a recipient id,
# and a value, specially calculated for each recipient, by taking the minimum of the incoming values 
# over all non-recipient ids, and multiplying by the product of signs (again, all signs other than the recipient sign)
    def __init__(self, identity, connectedVariableNodes):
        self.identity = identity
        numberOfConnectedVariableNodes = len(connectedVariableNodes)
        self.numberOfConnectedVariableNodes = numberOfConnectedVariableNodes
        self.incomingValues = np.zeros((numberOfConnectedVariableNodes,2), dtype = LDPC_DECIMAL_DATA_TYPE)
        self.incomingValues[:,0] = connectedVariableNodes
        
        self.outgoingValues = np.zeros((numberOfConnectedVariableNodes,2), dtype = LDPC_DECIMAL_DATA_TYPE)
        self.outgoingValues[:,0] = connectedVariableNodes

        self.signVector = np.ones(numberOfConnectedVariableNodes, dtype = LDPC_DECIMAL_DATA_TYPE)
        self.magnitudeVector = np.zeros(numberOfConnectedVariableNodes, dtype = LDPC_DECIMAL_DATA_TYPE)

        # Omer Sella: attempting to reduce decoder latency by removing the need of the function np.where
        self.variableIDtoIndexDictionary = typed.Dict.empty(*key_value_types)
        index = 0
        for cv in connectedVariableNodes:
            self.variableIDtoIndexDictionary[cv] = index
            index = index + 1
        self.mask = np.ones(self.numberOfConnectedVariableNodes, dtype = LDPC_BOOL_DATA_TYPE)
        
    def setSign(self):
        #Reset the sign vector from possible previous use
        self.signVector = self.signVector * 0 + 1
        #Set the sign of negatives to -1, positives and zero are left as 1
        self.signVector[ np.where(self.incomingValues[:,1] < 0) ] = -1
        sign = np.prod(self.signVector)
        return sign
        
    
    # Omer Sella: this function is replacing the receive function by using the dict data sturcture and removing np.where
    def receiveDictionary(self, variableID, value):
        index = self.variableIDtoIndexDictionary[variableID]
        self.incomingValues[index,1] = value
        return
    
    def receive(self, variableID, value):
        #print(self.incomingValues[:,0] == variableID)
        # Omer Sella: This is ultra-stupid, but it's a bug worth noting:
        # indexIn was created using np.where. Now, even though incomingValues[indexIn,0]
        # is a correct addressing and usage of indices (like MATLAB), the results of np.where is ([[someNumber]])
        # which is not an integer, and numba doesn't like this.
        #self.incomingValues[np.where(self.incomingValues[:,0] == variableID),1] = value - self.outgoingValues[np.where(self.outgoingValues[:,0] == variableID),1]
        indexIn = np.where(self.incomingValues[:,0] == variableID)[0][0]
        # Omer Sella: indexIn and indexOut are the same.
        #indexOut = np.where(self.outgoingValues[:,0] == variableID)[0][0]
        #assert(indexIn == indexOut)
        newValue = value - self.outgoingValues[indexIn,1]
        self.incomingValues[indexIn, 1] = newValue
        return
    
    def getValueForNodeDictionary(self, variableID):
        return self.outgoingValues[self.variableIDtoIndexDictionary[variableID],1] 
    
    
    def getValueForNode(self, variableID):
        value = self.outgoingValues[np.where(self.outgoingValues[:,0] == variableID)]
        value = value[0,1]
        return value

    def calcOutgoingValues(self):
        # Set the vector of signs (remember that the sign of 0 is 1), and obtain its product
        sign = self.setSign()
        # Set the vector of magnitudes using the numpy (standard) supplied abs function.
        self.magnitudeVector = np.abs(self.incomingValues[:,1])
        
        # Now we get cheeky: we locate the locations of the two lowest values in incoming values, say mindex_0 and m1.
        # They might be the same - we don't care.
        # Then we use just them for the outputs
        [m0,m1] = np.argsort(self.magnitudeVector)[0:2]
        smallest = self.magnitudeVector[m0]
        secondSmallest = self.magnitudeVector[m1]
        
        # Initialize outgoing values. Remember that sign is one of two options:  {1,-1}
        #self.outgoingValues[:,1] = self.outgoingValues[:,1] * 0 + sign
        #mask = np.ones(self.numberOfConnectedVariableNodes, dtype = LDPC_BOOL_DATA_TYPE)
        self.outgoingValues[:,1] = smallest * sign * self.signVector
        self.outgoingValues[m0,1] = secondSmallest * sign * self.signVector[m0]
        
        #for i in range(self.numberOfConnectedVariableNodes):
            # Set the mask to ignore the value at location i.
            #mask[i] = False
            # The following line should be read as if we are dividing rather than multiplying, but since we are dividing by either 1 or -1 it is the same as multiplyin. 
         #   self.outgoingValues[i,1] = self.outgoingValues[i,1] * self.signVector[i]
            # Once the sign was determined, we need to multiply by the minimum value, taken over all ABSOLUTE values EXCEPT at the i'th coordinate.
          #  self.outgoingValues[i,1] = self.outgoingValues[i,1] * np.min(self.magnitudeVector[mask])
            # Reset the mask to be the all True mask.
            #mask[i] = True
        return

        
specForLdpcDecoder = [
    ('H', NUMBA_FLOAT[:,:]),
    ('parityMatrix', NUMBA_FLOAT[:,:]),
    ('numberOfVariableNodes', NUMBA_INT),
    ('numberOfCheckNodes', NUMBA_INT),
    ('codewordLength', NUMBA_INT),
    ('softVector', NUMBA_FLOAT[:]),
    ('outgoingValues', NUMBA_FLOAT[:,:]),
    ('signVector', NUMBA_FLOAT[:]),
    ('magnitudeVector', NUMBA_FLOAT[:]),
    ('connectedVariableNodes', NUMBA_FLOAT[:]),
    ('variableNodes', types.ListType(variableNode)),
    ('checkNodes', types.ListType(checkNode)),
    ('checkNodeAddressBook', types.ListType(NUMBA_FLOAT)),
    ('variableNodeAddressBook', types.ListType(NUMBA_INT[:])),
    ('softVector', float64[:]),
    ]
#@jitclass(specForLdpcDecoder)
class ldpcDecoder:

    def __init__(self, H):
        self.parityMatrix = H
        m,n = H.shape
        self.numberOfVariableNodes = n
        self.numberOfCheckNodes = m
        self.codewordLength = n
        self.variableNodes = []
        self.checkNodes = []
        self.checkNodeAddressBook = []
        self.variableNodeAddressBook = []
        # Omer Sella: softVector is a place holder for the current state of a decoded vector (starting with information from the channel, and updating while iterating)
        self.softVector = np.zeros(self.numberOfVariableNodes, dtype = LDPC_DECIMAL_DATA_TYPE)
        for i in range(self.numberOfVariableNodes):
            addresses = np.where(self.parityMatrix[:,i] != 0)[0]
            self.variableNodeAddressBook.append(addresses)
            vn = variableNode(i)
            self.variableNodes.append(vn)
        for i in range(self.numberOfCheckNodes):
            # Omer Sella: Below there is a quick fix in the form of xyz[0]. It seems that the returned value is a sequence of length 1, where the first (and only) element is an array.
            addresses = np.where(self.parityMatrix[i,:] != 0)[0]
            self.checkNodeAddressBook.append(addresses)
            cn = checkNode(i, addresses)#self.checkNodeAddressBook[i])
            self.checkNodes.append(cn)
    
    def isCodeword(self, modulatedVector):
        binaryVector = slicer(modulatedVector, self.codewordLength)
        #Omer Sella: H.dot(binaryVector) is the same as summation over axis 0 of H[:,binaryVector] so we convert float multiplication into indexing and summation
        # Omer Sella: The following are equivalent (use of where, use of asarray(condition))
        #print(np.where(binaryVector != 0))
        #print(np.asarray(binaryVector != 0).nonzero())
        #result1 = self.parityMatrix.dot(binaryVector) % 2
        
        result2 = self.parityMatrix[:,np.asarray(binaryVector != 0).nonzero()[0]]
        #print(result2.shape)
        #np.bitwise_xor.reduce(result)
        result2 = np.sum(result2, axis = 1) % 2
        
        #assert np.all(result1 == result2)
        
        if all(result2 == 0):
            status = 'Codeword'
        else:
            status = 'Not a codeword'
        return status, binaryVector

    def decoderSet(self, fromChannel):
        for i in range(self.numberOfVariableNodes):
            self.variableNodes[i].fromChannel = fromChannel[i]
            self.variableNodes[i].presentState = fromChannel[i]
        return
    
    #def vn2cnTemp(self, j, i, value):
    #    self.checkNodes[j].receive(self.variableNodes[i].identity, value)
    #    #self.checkNodes[j].receiveDictionary(self.variableNodes[i].identity, value)
    #    #assert ( idx1 == idx2 )
    #    return

    def variableNodesToCheckNodes(self):
        for i in range(self.numberOfVariableNodes):
            value = self.variableNodes[i].presentState
            # Send the current value to all check nodes connected to this variable node.
            recipientCheckNodes = self.variableNodeAddressBook[i]
            for j in recipientCheckNodes:
                
                # Omer Sella: Two things I need to change: 1. isn't the identity simply i ? 2. Isn't there a way to send the index of the 
                    self.checkNodes[j].receive(self.variableNodes[i].identity, value)
            # Once the current value was sent, reset it to 0
            self.variableNodes[i].presentState = 0
        return
    
    def checkNodesToVariableNodes(self): 
        # We go over the check nodes one by one (possibly in parallel in the future) and calculate outgoing values.
        # Then whenever we have a check node that completed calculating outgoing values, have it broadcast these value to the corresponding nodes.
        for i in range(self.numberOfCheckNodes):
            self.checkNodes[i].calcOutgoingValues()
            outgoingValues = self.checkNodes[i].outgoingValues
            #print(outgoingValues)
            for k in range(len(outgoingValues[:,0])):
                self.variableNodes[LDPC_INT_DATA_TYPE(outgoingValues[k,0])].update(outgoingValues[k,1])
            #for j in self.checkNodeAddressBook[i]:
                #a = self.checkNodes[i].getValueForNodeDictionary(j)
                #b = self.checkNodes[i].getValueForNode(j)
                #assert(a == b)
                #self.variableNodes[j].update(b)
            #    self.variableNodes[j].update(self.checkNodes[i].getValueForNodeDictionary(j))

        return

    def decoderStep(self):
        # A decoder step is a broadcast from checknodes,
        # followed by a return broadcast from check nodes to variable nodes
        self.variableNodesToCheckNodes()
        self.checkNodesToVariableNodes()    
        # Reset the value of softVector
        softVector = self.softVector * 0
        # Finally we add the information from the channel to the present state and gather all present state values into a vector.
        for i in range(self.numberOfVariableNodes):
            self.variableNodes[i].presentState = self.variableNodes[i].presentState + self.variableNodes[i].fromChannel
            softVector[i] = self.variableNodes[i].presentState
        return softVector
        
    def decoderMainLoop(self, fromChannel, maxNumberOfIterations):
        status, binaryVector = self.isCodeword(fromChannel)
        softVector = np.copy(fromChannel)
        i = 0
        if status == 'Not a codeword':
            self.decoderSet(fromChannel)
            while (i < maxNumberOfIterations) & (status == 'Not a codeword'):
                i = i + 1
                softVector = self.decoderStep()
                status, binaryVector = self.isCodeword(softVector)
                #print('At iteration %d the status is: %s'%(i, status))
        return status, binaryVector, softVector, i



def testModulationAndSlicingRoundTrip():
    vectorLength = 100
    v = np.random.randint(0, 1, vectorLength, dtype = LDPC_DATA_TYPE)
    modulatedV = modulate(v, vectorLength)
    slicedV = slicer(modulatedV, vectorLength)
    assert (np.all(v == slicedV))
    return 'OK'


def evaluateCode(numberOfTransmissions, seed, SNRpoints, messageSize, codewordSize, numberOfIterations, H, G = 'None' ):
    # Concurrent futures require the seed to be between 0 and 2**32 -1
    #assert (np.dtype(seed) == np.int32)
    # note there is no usage of messageSize - keep it that way.
    assert (seed > 0)
    assert hasattr(SNRpoints, "__len__")
    localPrng = np.random.RandomState(seed)
    decoder = ldpcDecoder(H)
    numberOfSNRpoints = len(SNRpoints)
    
    # init a new berStatistics object to collect statistics
    berStats = common.berStatistics(codewordSize)#np.zeros(numberOfSNRpoints, dtype = LDPC_DECIMAL_DATA_TYPE)
    start = 0
    end = 0
    
    codeword = np.zeros(codewordSize, dtype = LDPC_INT_DATA_TYPE)
    decodedWord = np.zeros(codewordSize, dtype = LDPC_INT_DATA_TYPE)
    modulatedCodeword = modulate(codeword, codewordSize)    
    for s in range(numberOfSNRpoints):
        timeTotal = 0    
        SNR = SNRpoints[s]
        for k in range(numberOfTransmissions):
            
            dirtyModulated, sigma, sigmaActual = addAWGN(modulatedCodeword, codewordSize, SNR, localPrng) 
            senseword = slicer(dirtyModulated, codewordSize)            
            berUncoded = 0
            berDecoded = 0
            start = time.time()
            status, decodedWord, softVector, iterationStoppedAt = decoder.decoderMainLoop(dirtyModulated, numberOfIterations)
            end = time.time()
            timeTotal += (end - start)
            #print("******** " + str(np.sum(decodedWord == codeword)))
            berDecoded = np.count_nonzero(decodedWord != codeword)
            berStats.addEntry(SNR, sigma, sigmaActual, berUncoded, berDecoded, iterationStoppedAt, numberOfIterations, status)
        #print("Time it took the decoder:")
        #print(timeTotal)
        #print("And the throughput is:")
        numberOfBits = numberOfTransmissions * codewordSize
        #print(numberOfBits / timeTotal)
    return berStats

def evaluateCodeWrapper(seed, SNRpoints, numberOfIterations, parityMatrix, numOfTransmissions, G = 'None' , numberOfCores = 8):
    
    # This is a multiprocessing wrapper for evaluateCodeCuda.
    # No safety of len(seeds) == numberOfCudaDevices
    # No safety of cuda devices exist
    # Number of iterations must be divisible by numberOfCudaDevices
    localPRNG = np.random.RandomState(seed)
    seeds = localPRNG.randint(0, LDPC_MAX_SEED, numberOfCores, dtype = LDPC_SEED_DATA_TYPE)
    berStats = common.berStatistics()
    #It is assumed, i,.e., no safety, that the number of cores is => number of transmissions !!!!
    newNumOfTransmissions = numOfTransmissions // numberOfCores
    circulantSize = 511
    #circulantSize = 1021
    messageSize = 16 * circulantSize - (2 * circulantSize)
    codewordSize = 16 * circulantSize
    
    #Temporarily disabled for debug of cu_init error
    #print("*** debugging multiple futures. NumberOfCudaDevices: " + str(numberOfCudaDevices))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = {executor.submit(evaluateCode, newNumOfTransmissions, seeds[coreNumber], SNRpoints, messageSize, codewordSize, numberOfIterations, H = parityMatrix,  G = 'None'): coreNumber for coreNumber in range(numberOfCores)}
        #print(results)
    for result in concurrent.futures.as_completed(results):
        #print(result.result())
        berStats = berStats.add(result.result())
    return berStats



def evaluateCodeAtSingleTransmission(seed, SNRpoints, messageSize, codewordSize, numberOfIterations, H, G = 'None' ):
    # Concurrent futures require the seed to be between 0 and 2**32 -1
    #assert (np.dtype(seed) == np.int32)
    assert (seed > 0)
    assert hasattr(SNRpoints, "__len__")
    localPrng = np.random.RandomState(seed)
    decoder = ldpcDecoder(H)
    numberOfSNRpoints = len(SNRpoints)
    
    # init a new berStatistics object to collect statistics
    berStats = common.berStatistics()#np.zeros(numberOfSNRpoints, dtype = LDPC_DECIMAL_DATA_TYPE)
    for s in range(numberOfSNRpoints):
        berUncoded = 0
        berDecoded = 0
        SNR = SNRpoints[s]
        
        #print("*** transmission number " + str(transmission))
        ## loc == mean, scale == standard deviation (AKA sigma).
        if G == 'None':
            # Omer Sella: If G is not given we use the all 0 codeword, do not pass through message generation, do encode using multiplication by G.
            codeword = np.zeros(codewordSize, dtype = LDPC_INT_DATA_TYPE)
        else:
            message = localPrng.randint(0, 2, messageSize, dtype = LDPC_INT_DATA_TYPE)
            codeword = G.dot(message) % 2    
        modulatedCodeword = modulate(codeword, codewordSize)    
        dirtyModulated, sigma, sigmaActual = addAWGN(modulatedCodeword, codewordSize, SNR, localPrng) 
        dirtyModulated = copy.copy(modulatedCodeword)
        dirtyModulated[0] = dirtyModulated[0] * -1
        
        
        senseword = slicer(dirtyModulated, codewordSize)
        berUncoded = np.count_nonzero(senseword != codeword)
        start = time.time()
        status, decodedWord, softVector, iterationStoppedAt = decoder.decoderMainLoop(dirtyModulated, numberOfIterations)
        end = time.time()
        #print("******** " + str(np.sum(decodedWord == codeword)))
        berDecoded = np.count_nonzero(decodedWord != codeword)
        berStats.addEntry(SNR, sigma, sigmaActual, berUncoded, berDecoded, iterationStoppedAt, numberOfIterations, status)
        
    return berStats

def constantFunction(const):
    def g(x):
        return const
    return g


def testCodeUsingMultiprocessing(seed, SNRpoints, messageSize, codewordSize, numberOfIterations, numberOfTransmissions, H, method = None, reference = None, G = 'None'):
    bStats = common.berStatistics(codewordSize)
    localPrng = np.random.RandomState(seed)
    seeds = localPrng.randint(0, LDPC_MAX_SEED, numberOfTransmissions, dtype = LDPC_SEED_DATA_TYPE) 
    
    mesL = [messageSize] * numberOfTransmissions
    cwsL = [codewordSize] * numberOfTransmissions
    itrL = [numberOfIterations] * numberOfTransmissions
    hL = [H] * numberOfTransmissions
    if method == None:
        #Omer Sella: This is a cheasy fix, and only works for the constant function. Need to generalise this.
        #g = constantFunction(reference)
        
        snrL = [SNRpoints] * numberOfTransmissions
        for s in SNRpoints:
            sL = [[s]] * numberOfTransmissions
            start = time.time()
            with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            # Omer Sella: we are using multiprocessing and distributing over TRANSMISSIONS (which we expect to have O(numberOfTransmissions)) rather than SNR points (which we expect to have O(3)).
                results = executor.map(evaluateCodeAtSingleTransmission, seeds, sL, mesL, cwsL, itrL, hL)
                for r in results:
                    bStats = bStats.union(r)
            end = time.time()
            print(" Time it took the decoder at snr "+ str(s) + " is:")
            print(end-start)
            print("And the throughput is: ")
            print(numberOfTransmissions * codewordSize / (end-start))
    else:
        for s in SNRpoints:
            snrL = [[s]] * numberOfTransmissions
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = executor.map(evaluateCodeAtSingleTransmission, seeds, snrL, mesL, cwsL, itrL, hL)
            for r in results:
                bStats = bStats.union(r)
            snrAxis, averageSnrAxis, berData, averageNumberOfIterations = bStats.getStats(messageSize)
            print(berData)
            if (berData[0] > reference): 
                print("*** Evaluation failed at " + str(len(bStats.snrAxis)))
                break
    return bStats



def testNearEarth(numOfTransmissions = 60, cores = 1):
    print("*** in test near earth")
    nearEarthParity = fileHandler.readMatrixFromFile(str(projectDir) + '/codeMatrices/nearEarthParity.txt', 1022, 8176, 511, True, False, False)
    #numOfTransmissions = 50
    roi = [3.0, 3.2, 3.4]#,3.6]#np.arange(3, 3.8, 0.2)
    codewordSize = 8176
    messageSize = 7154
    numOfIterations = 50

    start = time.time()
    
    #bStats = evaluateCode(numOfTransmissions, 460101, roi, messageSize, codewordSize, numOfIterations, nearEarthParity)    
    #for i in range(numOfTransmissions):
    #    bStats = evaluateCodeAtSingleTransmission(460101, roi, messageSize, codewordSize, numOfIterations, nearEarthParity)    
        
    bStats = testCodeUsingMultiprocessing(460101, roi, messageSize, codewordSize, numOfIterations, numOfTransmissions, nearEarthParity)
    end = time.time()
    print('****Time it took for code evaluation == %d' % (end-start))
    print('****Throughput == '+str((8176*len(roi)*numOfTransmissions)/(end-start)) + 'bits per second.')
    #a, b, c, d = bStats.getStats(codewordSize)
    #print("berDecoded " + str(c))
    return bStats

def testBeast(numOfTransmissions = 16*1):
    print("*** testing beast")
    beastParity = fileHandler.readMatrixFromFile(str(projectDir) + '/codeMatrices/nearEarthParity.txt', 2042, 16336, 1021, True, False, False)
    #numOfTransmissions = 50
    roi = [3.0, 3.2, 3.4]
    codewordSize = 16336
    messageSize = (16336 - 2042)
    numOfIterations = 50

    start = time.time()
    
    #bStats = evaluateCode(numOfTransmissions, 460101, roi, messageSize, codewordSize, numOfIterations, nearEarthParity)    
    #for i in range(numOfTransmissions):
    #    bStats = evaluateCodeAtSingleTransmission(460101, roi, messageSize, codewordSize, numOfIterations, nearEarthParity)    
    #bStats = evaluateCode(numberOfTransmissions = numOfTransmissions, seed = 460101, SNRpoints = roi, messageSize = messageSize, codewordSize = codewordSize, numberOfIterations = numOfIterations, H = beastParity, G = 'None' )
    bStats = evaluateCodeWrapper(seed = 460101, SNRpoints = roi, numberOfIterations = numOfIterations, numOfTransmissions = numOfTransmissions, parityMatrix = beastParity, numberOfCores = 16)
    end = time.time()
    print('****Time it took for code evaluation == %d' % (end-start))
    print('****Throughput == '+str((16336*len(roi)*numOfTransmissions)/(end-start)) + 'bits per second.')
    #a, b, c, d = bStats.getStats(codewordSize)
    #print("berDecoded " + str(c))
    return bStats


def testWifi(numOfTransmissions = 50):
    print("*** test wifi is decomissioned !!!!")
    wifiParity = wifiMatrices.getWifiParityMatrix()
    #numOfTransmissions = 50
    roi = [3.0, 3.2,3.4,3.6]#np.arange(3, 3.8, 0.2)
    codewordSize = 1944
    messageSize = 1620
    numOfIterations = 50

    start = time.time()
    
    #bStats = evaluateCode(numOfTransmissions, 460101, roi, messageSize, codewordSize, numOfIterations, wifiParity)    
    #for i in range(numOfTransmissions):
    #    bStats = evaluateCodeAtSingleTransmission(460101, roi, messageSize, codewordSize, numOfIterations, nearEarthParity)    
        
    #bStats = testCodeUsingMultiprocessing(460101, roi, messageSize, codewordSize, numOfIterations, numOfTransmissions, nearEarthParity)
    end = time.time()
    print('Time it took for code evaluation == %d' % (end-start))
    print('Throughput == '+str((8176*len(roi)*numOfTransmissions)/(end-start)) + 'bits per second.')
    #a, b, c, d = bStats.getStats(codewordSize)
    #print("berDecoded " + str(c))
    return bStats

# Omer Sella: Name guarding is needed when doing concurrent futures in Windows (i.e.: if __name__ ...)
def main():
    print("*** In ldpc.py main function.")
    #bStats = testWifi()
    #bStats = testNearEarth()
    seed = 7134066
    nearEarthParity = fileHandler.readMatrixFromFile(str(projectDir) + '/codeMatrices/nearEarthParity.txt', 1022, 8176, 511, True, False, False)
    #numOfTransmissions = 50
    roi = [3.0, 3.2, 3.4]#,3.6]#np.arange(3, 3.8, 0.2)
    codewordSize = 8176
    messageSize = 7154
    numOfIterations = 50
    numOfTransmissions = 56

    for i in [1,2,4,7,8, 14, 28, 56]:
        start = time.time()
        bStats = evaluateCodeWrapper(seed = seed, SNRpoints = roi, numberOfIterations = numOfIterations, parityMatrix = nearEarthParity, numOfTransmissions = numOfTransmissions, G = 'None' , numberOfCores = i)
        end = time.time()
        print('Time it took for code evaluation == %d' % (end-start))
        print('Throughput == '+str((8176*len(roi)*numOfTransmissions)/(end-start)) + 'bits per second.')
    #bStats = testBeast()
    #scatterSNR, scatterBER, scatterITR, snrAxis, averageSnrAxis, berData, averageNumberOfIterations = bStats.getStatsV2(16336)
    #common.plotEvaluationData(scatterSNR, scatterBER)
    return bStats
    
if __name__ == '__main__':
    bStats = main()


#### Deprecated:
#    def ldpcTestNearEarthSingleTransmissionUsingSeed(seed):
#    pointsOfInterest = [3.2, 3.21, 3.22, 3.23, 3.24, 3.25]#, 3.26, 3.27, 3.28, 3.29, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0]
#    messageSize = 7154
#    codewordSize = 8176
#    numberOfIterations = 50
#    #nearEarthGenerator = fileHandler.readMatrixFromFile(projectDir + '/codeMatrices/nearEarthGenerator.txt', 7154, 8176, 511, True, True, True)
#    nearEarthParity = fileHandler.readMatrixFromFile(projectDir + '/codeMatrices/nearEarthParity.txt', 1022, 8176, 511, True, False, False)
#    #nearEarthGeneratorT = nearEarthGenerator.T
#    return evaluateCodeAtSingleTransmission(seed, pointsOfInterest, messageSize, codewordSize, numberOfIterations, nearEarthParity)
#
#def zeroCodewordAWGNSimulation(H, pointsOfInterest, numberOfIterations, numberOfTransmissions, seed):
#    start = time.time()
#    localPRNG = np.random.RandomState(seed)
#    decoder = ldpcDecoder(H)
#    numberOfParityChecks,messageSize = H.shape
#    #fileName = "c:/Users/Omer/swift/decoderDubug.txt"
#    #fid = open(fileName, "w")
#    bStats = common.berStatistics()
#    for k in range(len(pointsOfInterest)):
#        SNR = pointsOfInterest[k]
#        print("*** checking SNR == " + str(SNR))
#        for transmission in range(numberOfTransmissions):
#            print("*** transmission number " + str(transmission))
#            ## loc == mean, scale == standard deviation (AKA sigma).
#            message = np.zeros(messageSize, dtype = LDPC_INT_DATA_TYPE)
#            modulatedCodeword = modulate(message, messageSize)
#            seed = LDPC_LOCAL_PRNG.randint(0, 2**31 - 1, dtype = np.int32)
#            dirtyModulated, sigma, sigmaActual = addAWGN(modulatedCodeword, messageSize, SNR, localPRNG)
#            uncodedErrors = np.count_nonzero(slicer(dirtyModulated, messageSize) != message)
#            status, decodedWord, softVector, itr = decoder.decoderMainLoop(dirtyModulated, numberOfIterations)
#            decodedErrors = np.count_nonzero(decodedWord != message)
#            #fid.write(str(SNR) + " " + str(sigma)+ " " + str(sigmaActual) + " " + str(status) + " " + str(itr) + " " + str(uncodedErrors) + " " + str(np.count_nonzero(decodedWord != message)) + "\n")
#            bStats.addEntry(SNR, sigma, sigmaActual, uncodedErrors, decodedErrors, itr, numberOfIterations, status)
#    #fid.close()
#    end = time.time()
#    fileHandler.saveCodeInstance(H, 511, 8176, bStats, projectDirEvals)
#    print("Time it took to evaluate code: " + str(end - start))
#    return bStats
