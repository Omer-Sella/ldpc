# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:10:11 2019

@author: Omer
"""

## File handler
## This file was initially intended purely to generate the matrices for the near earth code found in: https://public.ccsds.org/Pubs/131x1o2e2s.pdf
## The values from the above pdf were copied manually to a txt file, and it is the purpose of this file to parse it.
## The emphasis here is on correctness, I currently do not see a reason to generalise this file, since matrices will be saved in either json or some matrix friendly format.

import numpy as np
from scipy.linalg import circulant
#import matplotlib.pyplot as plt
import scipy.io
import common
import hashlib
import os


FILE_HANDLER_INT_DATA_TYPE = np.int32
GENERAL_CODE_MATRIX_DATA_TYPE = np.int32
NIBBLE_CONVERTER = np.array([8, 4, 2, 1], dtype = GENERAL_CODE_MATRIX_DATA_TYPE)



def nibbleToHex(inputArray):
    n = NIBBLE_CONVERTER.dot(inputArray)
    if n == 10:
        h = 'A'
    elif n== 11:
        h = 'B'
    elif n== 12:
        h = 'C'
    elif n== 13:
        h = 'D'
    elif n== 14:
        h = 'E'
    elif n== 15:
        h = 'F'
    else:
        h = str(n)
    return h
    
def binaryArraytoHex(inputArray):
    d1 = len(inputArray)
    assert (d1 % 4 == 0)
    
    outputArray = np.zeros(d1//4, dtype = str)
    outputString = ''
    for j in range(d1//4):
        nibble = inputArray[4 * j : 4 * j + 4]
        h = nibbleToHex(nibble)
        outputArray[j] = h
        outputString = outputString + h
    return outputArray, outputString
   

def hexStringToBinaryArray(hexString):
    outputBinary = np.array([], dtype = GENERAL_CODE_MATRIX_DATA_TYPE)
    for i in hexString:
        if i == '0':
            nibble =  np.array([0,0,0,0], dtype = GENERAL_CODE_MATRIX_DATA_TYPE)
        elif i ==  '1':
            nibble =  np.array([0,0,0,1], dtype = GENERAL_CODE_MATRIX_DATA_TYPE)
            
        elif i ==  '2':
            nibble =  np.array([0,0,1,0], dtype = GENERAL_CODE_MATRIX_DATA_TYPE)
            
        elif i ==  '3':
            nibble =  np.array([0,0,1,1], dtype = GENERAL_CODE_MATRIX_DATA_TYPE)
            
        elif i ==  '4':
            nibble =  np.array([0,1,0,0], dtype = GENERAL_CODE_MATRIX_DATA_TYPE)
            
        elif i ==  '5':
            nibble =  np.array([0,1,0,1], dtype = GENERAL_CODE_MATRIX_DATA_TYPE)
            
        elif i ==  '6':
            nibble =  np.array([0,1,1,0], dtype = GENERAL_CODE_MATRIX_DATA_TYPE)
            
        elif i ==  '7':
            nibble =  np.array([0,1,1,1], dtype = GENERAL_CODE_MATRIX_DATA_TYPE)
            
        elif i ==  '8':
            nibble =  np.array([1,0,0,0], dtype = GENERAL_CODE_MATRIX_DATA_TYPE)
            
        elif i ==  '9':
            nibble =  np.array([1,0,0,1], dtype = GENERAL_CODE_MATRIX_DATA_TYPE)
            
        elif i ==  'A':
            nibble =  np.array([1,0,1,0], dtype = GENERAL_CODE_MATRIX_DATA_TYPE)
            
        elif i ==  'B':
            nibble =  np.array([1,0,1,1], dtype = GENERAL_CODE_MATRIX_DATA_TYPE)
            
        elif i ==  'C':
            nibble =  np.array([1,1,0,0], dtype = GENERAL_CODE_MATRIX_DATA_TYPE)
            
        elif i ==  'D':
            nibble =  np.array([1,1,0,1], dtype = GENERAL_CODE_MATRIX_DATA_TYPE)
            
        elif i ==  'E':
            nibble =  np.array([1,1,1,0], dtype = GENERAL_CODE_MATRIX_DATA_TYPE)
            
        elif i ==  'F':
            nibble =  np.array([1,1,1,1], dtype = GENERAL_CODE_MATRIX_DATA_TYPE)
            
        else:
            #print('Error, 0-9 or A-F')
            pass
            nibble = np.array([], dtype = GENERAL_CODE_MATRIX_DATA_TYPE)
        outputBinary = np.hstack((outputBinary, nibble))
    return outputBinary
            

def hexToCirculant(hexStr, circulantSize):
    binaryArray = hexStringToBinaryArray(hexStr)
    
    if len(binaryArray) < circulantSize:
        binaryArray = np.hstack(np.zeros(circulantSize-len(binaryArray), dtype = GENERAL_CODE_MATRIX_DATA_TYPE))
    else:
        binaryArray = binaryArray[1:]
    circulantMatrix = circulant(binaryArray)
    circulantMatrix = circulantMatrix.T
    return circulantMatrix

def hotLocationsToCirculant(locationList, circulantSize):
    generatingVector = np.zeros(circulantSize, dtype = GENERAL_CODE_MATRIX_DATA_TYPE)
    generatingVector[locationList] = 1
    newCirculant = circulant(generatingVector)
    newCirculant = newCirculant.T
    return newCirculant

def readMatrixFromFile(fileName, dim0, dim1, circulantSize, isRow = True, isHex = True, isGenerator = True ):
    # This function assumes that each line in the file contains the non zero locations of the first row of a circulant.
    # Each line in the file then defines a circulant, and the order in which they are defined is top to bottom left to right, i.e.:
    # line 0 defines circulant 0,0
    
    with open(fileName) as fid:
        lines = fid.readlines()    
    if isGenerator:
        for i in range((dim0 // circulantSize)  ):
            bLeft = hexToCirculant(lines[2 * i], circulantSize)
            bRight = hexToCirculant(lines[2 * i + 1], circulantSize)
            newBlock = np.hstack((bLeft, bRight))
            if i == 0:
                accumulatedBlock = newBlock
            else:
                accumulatedBlock = np.vstack((accumulatedBlock, newBlock))
        newMatrix = np.hstack((np.eye(dim0, dtype = GENERAL_CODE_MATRIX_DATA_TYPE), accumulatedBlock))
    else:
        
        for i in range((dim1 // circulantSize)):
            locationList1 = list(lines[ i].rstrip('\n').split(','))
            locationList1 = list(map(int, locationList1))
            upBlock = hotLocationsToCirculant(locationList1, circulantSize)
            if i == 0:
                accumulatedUpBlock1 = upBlock
            else:
                accumulatedUpBlock1 = np.hstack((accumulatedUpBlock1, upBlock))
        
        for i in range((dim1 // circulantSize)):
            locationList = list(lines[(dim1 // circulantSize) + i].rstrip('\n').split(','))
            locationList = list(map(int, locationList))
            newBlock = hotLocationsToCirculant(locationList, circulantSize)
            if i == 0:
                accumulatedBlock2 = newBlock
            else:
                accumulatedBlock2 = np.hstack((accumulatedBlock2, newBlock))
        newMatrix = np.vstack((accumulatedUpBlock1, accumulatedBlock2))
    return newMatrix
            
def binaryMatrixToHexString(binaryMatrix, circulantSize):
    
    leftPadding = np.array(4 - (circulantSize % 4))
    m,n = binaryMatrix.shape
    #print(m)
    #print(n)
    assert( m % circulantSize == 0)
    assert (n % circulantSize == 0)
    
    M = m // circulantSize
    N = n // circulantSize
    hexName = ''
    for r in range(M):
        for k in range(N):
            nextLine = np.hstack((leftPadding, binaryMatrix[ r * circulantSize , k * circulantSize : (k + 1) * circulantSize]))
            hexArray, hexString = binaryArraytoHex(nextLine)
            hexName = hexName + hexString
    return hexName


def saveCodeInstance(parityMatrix, circulantSize, codewordSize, evaluationData, path, evaluationTime, numberOfNonZero):
    print("*** in saveCodeInstance ...")
    m, n = parityMatrix.shape
    M = m // circulantSize
    N = n // circulantSize
    fileName = binaryMatrixToHexString(parityMatrix, circulantSize)
    fileNameSHA224 = str(circulantSize) + '_' + str(M) + '_' + str(N) + '_' + str(hashlib.sha224(str(fileName).encode('utf-8')).hexdigest())
    fileNameWithPath = path + fileNameSHA224
    workspaceDict = {}
    workspaceDict['parityMatrix'] = parityMatrix
    workspaceDict['fileName'] = fileName
    scatterSNR, scatterBER, scatterITR, snrAxis, averageSnrAxis, berData, averageNumberOfIterations = evaluationData.getStatsV2()
    workspaceDict['snrData'] = scatterSNR
    workspaceDict['berData'] = scatterBER
    workspaceDict['itrData'] = scatterITR
    workspaceDict['averageSnrAxis'] = averageSnrAxis
    workspaceDict['averageNumberOfIterations'] = averageNumberOfIterations
    workspaceDict['evaluationTime'] = evaluationTime
    workspaceDict['nonZero'] = numberOfNonZero
    scipy.io.savemat((fileNameWithPath + '.mat'), workspaceDict)
    #evaluationData.plotStats(codewordSize, fileNameWithPath)
    print("*** Finishing saveCodeInstance !")
    return fileName

def testFileHandler():
    nearEarthGenerator = readMatrixFromFile('/codeMatrices/nearEarthGenerator.txt', 7154, 8176, 511, True, True, True)
    nearEarthParity = readMatrixFromFile('/codeMatrices/nearEarthParity.txt', 1022, 8176, 511, True, False, False)
    return 'OK'


def plotResults(path, makeMat = False):
    i = 10
    evaluationFaildAt = np.zeros(4, dtype = FILE_HANDLER_INT_DATA_TYPE)
    evalTimes = []
    numberOfIterationsAtHigh = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if str(file).endswith('.mat'):
                i = i + 1
                mat = scipy.io.loadmat(str(os.path.join(root, file)))
                snrAxis = mat['snrAxis']
                snrActual = mat['averageSnrAxis']
                if len(snrAxis) < 3:
                    evaluationFaildAt[len(snrAxis)] = evaluationFaildAt[len(snrAxis)] + 1
                berAxis = mat['berData']
                if ('evaluationTime' in mat.keys()):
                    evalTimes.append(mat['evaluationTime'])
                averageNumberOfIterations = mat['averageNumberOfIterations']
                numberOfIterationsAtHigh.append(averageNumberOfIterations[-1])
                common.plotSNRvsBER(snrActual, berAxis, fileName = None, inputLabel = '', figureNumber = i, figureName = str(file))
            else:
                pass
    return evalTimes, evaluationFaildAt, numberOfIterationsAtHigh
        
        
#plt.imshow(nearEarthParity)
    

#nearEarthParity = readMatrixFromFile('/home/oss22/swift/swift/codeMatrices/nearEarthParity.txt', 1022, 8176, 511, True, False, False)
#import networkx as nx
#from networkx.algorithms import bipartite
#B = nx.Graph()
#B.add_nodes_from(range(1022), bipartite=0)
#B.add_nodes_from(range(1022, 7156 + 1022), bipartite=1)
# Add edges only between nodes of opposite node sets

#for i in range(8176):
#    for j in range(1022):
#        if nearEarthParity[j,i] != 0:
#            B.add_edges_from([(j, 7156 + i)])

#X, Y = bipartite.sets(B)
#pos = dict()
#pos.update( (n, (1, i)) for i, n in enumerate(X) )
#pos.update( (n, (2, i)) for i, n in enumerate(Y) )
#nx.draw(B, pos=pos)
#plt.show()



