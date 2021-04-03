# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:03:18 2020

@author: Omer
"""
### New encoder / decoder implementation using tensorflow
import tensorflow as tf
import numpy as np
import os

projectDir = os.environ.get('SWIFT')
if projectDir == None:
    import pathlib
    projectDir = pathlib.Path(__file__).parent.absolute()

#projectDir = "/local/scratch/oss22/swift/"
#projectDir = "c:/Users/Omer/swift/"
projectDirEvals = projectDir + "evaluations/"

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, projectDir)

import fileHandler
import common

## Stages of the decoder:

# 1. Take the channel input and blow up from 1D to 2D
# 2. Multiply elemntwise by H, i.e.: use H as a mask.
# 3. Blow up the result (so now 3D).
# 4. Create a mask that has +inf for every columnn the 2D version.
# 5. Add the latter mask to the 3D blowup.
# 6. Create two reductions: one by minimum and one by multiplication of sign.
# 7. Multiply elementwise the two reductions.
# 8. Now multiply elementwise with H again.
# 9. Now reduce using summation over VNs


# Constants
LDPC_TF_FLOAT32 = tf.dtypes.float32 
LDPC_TF_VARIABLE_NODE_AXIS = 1
LDPC_TF_USE_SPARSE_MATRICES = False
LDPC_TF_INT_64 = tf.dtypes.int64
LDPC_NP_INT_32_TYPE = np.int32
LDPC_NP_FLOAT_64_TYPE = np.float64
LDPC_TF_INF = np.inf # np.iinfo(LDPC_NP_INT_32_TYPE).max
LDPC_TF_LOCAL_PRNG = np.random.RandomState(7134066)


def addAWGN(message, snrDB, prng):
    snr = 10 ** (snrDB / 10)
    sigma = tf.math.sqrt(snr)
    noise = prng.normal(0.0, sigma, tf.shape(message))#dtype=LDPC_NP_INT_32_TYPE)
    noiseSumSquares = tf.math.reduce_sum(tf.square(noise))
    howMany = LDPC_NP_FLOAT_64_TYPE(tf.math.reduce_sum(tf.shape(noise)))
    sigmaActual = tf.math.sqrt( noiseSumSquares / howMany)
    noisyMessage = message + noise
    return noisyMessage, sigma, sigmaActual


def slicer(message):
    result = tf.where(message > 0, 1, 0)
    return result

def modulate(message):
    result = tf.where(message == 0, -1, message)
    return result


def variableNodeStep(tensorOfChecks):
    result = tf.reduce_sum(tensorOfChecks, LDPC_TF_VARIABLE_NODE_AXIS)
    return result


#Omer Sella: when blowUpdDimIs0 == True, this function takes a k-dimensional tensor and a scalar b, and returns a (k+1) dimensional tensor, with b copies of the origin along the *0* dimension.
def blowUp(origin, blowUpFactor, blowUpDimIs0 = True):
    originShape = origin.shape
    tilingParams = np.ones(len(originShape), dtype = LDPC_NP_INT_32_TYPE)
    if blowUpDimIs0 == True:    
        tilingParams[0] = blowUpFactor
        result = tf.tile(origin, tilingParams)
        newShape = tf.concat([[blowUpFactor], originShape], axis = 0)
    else:
        
        tilingParams[len(tilingParams) - 1] = blowUpFactor
        result = tf.tile(origin, tilingParams)
        newShape = tf.concat([originShape, [blowUpFactor]], axis = 0)
    result = tf.reshape(result, newShape)
    return result

def oneInfColumn3D(checkDim, variableDim):   
    hotOneColumn = tf.reverse(tf.eye(variableDim, dtype = LDPC_TF_FLOAT32), [0])
    hotOneColumn = tf.tile(hotOneColumn, [checkDim, 1])
    hotOneColumn = tf.reshape(hotOneColumn, [checkDim, variableDim, variableDim])
    oneInf3d = tf.where(hotOneColumn == 1, LDPC_TF_INF, 0)
    return oneInf3d


def minSumStep(H, fromChannel):
    checkDim, variableDim = H.shape
    # 1. Take the channel input and blow up from 1D to 2D
    fromChannelBlown = blowUp(fromChannel, checkDim)
    #print(fromChannelBlown)
    # 2. Multiply elemntwise by H, i.e.: use H as a mask.
    result = tf.math.multiply(H, fromChannelBlown)
    #print(result)
    # 3. Blow up the result (so now 3D).
    result = blowUp(result, variableDim, False)
    result = np.float32(result)
    # 4. Create a mask that has +inf for every columnn the 2D version.
    oneInf3d = oneInfColumn3D(checkDim, variableDim)
    # 5. Add the latter mask to the 3D blowup.
    result = result + oneInf3d
    # 6. Create two reductions: one by minimum and one by multiplication of sign.
    #print(result.dtype)
    #print(result)
    SAME_AXIS = 1
    minimumReduction = tf.reduce_min(result, axis = SAME_AXIS)
    #print(minimumReduction)
    signMultiplicationReduction = tf.reduce_prod( tf.sign(result), axis = SAME_AXIS )
    #print(signMultiplicationReduction)
    # 7. Multiply elementwise the two reductions.
    reduced = tf.math.multiply(minimumReduction, signMultiplicationReduction)
    # 8. Now multiply elementwise with H again.
    reduced = tf.math.multiply(reduced, H)
    # 9. Now reduce using summation over VNs, and add 
    sumReduced = tf.reduce_sum(reduced, axis = 0)
    return sumReduced


def isCodeword(H, message):
    H_new = tf.cast(H, tf.float32)
    result = tf.linalg.matvec(H_new, message, a_is_sparse=LDPC_TF_USE_SPARSE_MATRICES )
    result = result % 2
    nnz = tf.math.count_nonzero(result, dtype=LDPC_TF_INT_64)
    nnz = nnz.numpy()
    if ( nnz == 0):
        status = 'Codeword'
    else:
        status = 'Not a codeword'
    return status, result

def decoderMainLoop(H, fromChannel, maxNumberOfIterations):
        fromChannel = np.float32(fromChannel)
        status, binaryVector = isCodeword(H, fromChannel)
        presentState = np.copy(fromChannel)
        i = 0
        if status != 'Codeword':
            while (i < maxNumberOfIterations) & (status == 'Not a codeword'):
                i = i + 1
                update = minSumStep(H, presentState)
                print(update)
                print(presentState)
                presentState = presentState + update
                status, binaryVector = isCodeword(H, presentState)
                print('At iteration %d the status is: %s'%(i, status))
        else:
            print('Codeword found, no iterations needed')
        return status, binaryVector, presentState, i

def testMinSumStep():
    H = np.random.randint(0,2,[10,15])
    fc = np.float32(np.random.randint(0,2,15))
    result = minSumStep(H, fc)
    return 'OK'

def testModulationAndSlicingRoundtrip():
    vector = np.random.randint(0,2,100)
    modulatedVector = modulate(vector)
    slicedVector = slicer(modulatedVector)
    if np.all(vector == slicedVector):
        status = 'OK'
    else:
        status = 'Modulation and slicing error'
    return status

def ldpcTFTestNearEarth(seed = 7134066):
    pointsOfInterest = [3.0, 3.2,3.4,3.6]#, 3.26, 3.27, 3.28, 3.29, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0]
    numberOfIterations = 50
    nearEarthParity = fileHandler.readMatrixFromFile(projectDir + '/codeMatrices/nearEarthParity.txt', 1022, 8176, 511, True, False, False)
    nearEarthParity = nearEarthParity[0:10,:]
    return evaluateCodeAtSingleTransmission(seed, pointsOfInterest, numberOfIterations, nearEarthParity)

def evaluateCodeAtSingleTransmission(seed, SNRpoints, numberOfIterations, H, G = 'None' ):
    assert (seed > 0)
    assert hasattr(SNRpoints, "__len__")
    checkSize, codewordSize = H.shape
    #print(H.shape)
    numberOfSNRpoints = len(SNRpoints)
    
    # init a new berStatistics object to collect statistics
    berStats = common.berStatistics()#np.zeros(numberOfSNRpoints, dtype = LDPC_DECIMAL_DATA_TYPE)
    for s in range(numberOfSNRpoints):
        SNR = SNRpoints[s]
        #print("*** transmission number " + str(transmission))
        ## loc == mean, scale == standard deviation (AKA sigma).
        # Omer Sella: If G is not given we use the all 0 codeword, do not pass through message generation, do encode using multiplication by G.
        zeroCodeword = np.zeros(codewordSize, dtype = LDPC_NP_INT_32_TYPE)
        modulatedCodeword = modulate(zeroCodeword)    
        dirtyModulated, sigma, sigmaActual = addAWGN(modulatedCodeword, SNR, LDPC_TF_LOCAL_PRNG) 
        senseword = slicer(dirtyModulated)
        #assert (len(codeword) == len(senseword))
        berUncoded = np.count_nonzero(senseword != zeroCodeword)
        status, decodedWord, softVector, iterationStoppedAt = decoderMainLoop(H, dirtyModulated, numberOfIterations)
        berDecoded = np.count_nonzero(decodedWord != zeroCodeword)
        berStats.addEntry(SNR, sigma, sigmaActual, berUncoded, berDecoded, iterationStoppedAt, numberOfIterations, status)
    return berStats

def main():
    print("*** In ldpcTF.py main function.")
    bStats = ldpcTFTestNearEarth()
    return bStats
    
if __name__ == '__main__':
    bStats = main()
