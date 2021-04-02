# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:26:10 2021

@author: Omer Sella
"""
import numpy as np
import time
import os
import mpiFunctions
from mpiFunctions import mpiProcessID
import h5py

PROJECT_PATH = os.environ.get('LDPC')

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def numToBits(number, numberOfBits):
    assert number < 16
    assert number >= 0
    newNumber = np.zeros(numberOfBits, dtype = int)
    for j in range(numberOfBits - 1, -1, -1):
        newNumber[j] = newNumber[j] + (number % 2)
        number = number >> 1
    return newNumber

#class plotter():
    
#    def __init__(self, axisPairs):
        

def colourString(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

class logger():
    
    def __init__(self, keys, agentSeed = None, environmentSeed = None, logPath = None, hdf5FileName = 'experiment.h5', fileName = 'experiment.txt', graphics = False):
        
        
        if  mpiProcessID() == 0:
            if logPath == None:
                self.logPath = PROJECT_PATH + "/temp/experiments/%i" %int(time.time())
            else:
                self.logPath = logPath
            if os.path.exists(self.logPath):
                print("Warning: Log dir %s already exists."%self.logPath)
            else:
                os.makedirs(self.logPath)
            self.fileName = os.path.join(self.logPath, fileName)
            self.hdf5FileName = os.path.join(self.logPath, hdf5FileName)
        else:
            self.logPath = None
            self.fileName = None
            self.hdf5FileName = None
        self.currentRow = {}
        self.columnKeys = []
        if agentSeed != None:
            self.environmentSeed = environmentSeed
        if environmentSeed != None:
            self.agentSeed = agentSeed
        for key in keys:
            self.columnKeys.append(key)
        if mpiProcessID() == 0:            
            with open(self.fileName, 'w') as fid:
                fid.write("\t".join(self.columnKeys)+"\n")
            with h5py.File(self.hdf5FileName, 'a') as fid:
                for key in self.columnKeys:
                    fid.create_group(key)
        self.dataSet = 0
        
    def logPrint(self, message, colour='green'):
        if mpiProcessID() == 0:
            print(colourString(message, colour, bold = True))
    
    def keyValue(self, key, value):
        assert key in self.columnKeys, "Assertion failed since the key %s in keyValue(key, value) was not introduced when the logger was initialised."%key
        assert key not in self.currentRow, "Assertion failed since the value for key %s already exists in this row"%key
        self.currentRow[key] = value
        return 'OK'
        
    def dumpLogger(self):
        if mpiProcessID() == 0:
            values = []
            keyLengths = []
            for key in self.columnKeys:
                keyLengths.append(len(key))
            maximalKeyLength = max(15,max(keyLengths))
            keyString = '%'+'%d'%maximalKeyLength
            stringFormat = "| " + keyString + "s | %15s |"
            numberOfDashes = 22 + maximalKeyLength
            print("-"*numberOfDashes)
            for key in self.columnKeys:
                value = self.currentRow.get(key, "")
                if isinstance(value, np.ndarray):
                    valueString = value
                elif hasattr(value, "__float__"):
                    valueString = "%8.3g"%value
                else:
                    valueString = value
                print(stringFormat%(key, valueString))
                values.append(value)
            print("-"*numberOfDashes, flush=True)
            if self.fileName is not None:
                with open(os.path.join(self.logPath, self.fileName), 'a') as fid:
                    fid.write("\t".join(map(str,values))+"\n")
                    fid.flush()
                with h5py.File(self.hdf5FileName, 'a') as fid:
                    fid.create_group(str(self.dataSet))
                    for key in self.columnKeys:
                        fid[str(self.dataSet)].attrs[key] = self.currentRow[key]
                        fid[key].attrs[str(self.dataSet)] = self.currentRow[key]
                    self.dataSet = self.dataSet + 1
                self.currentRow.clear()

def testLogger():
    status = 'OK'
    keys = ['minimum', 'maximum', 'average', 'serialNumber']
    myLogger = logger(keys)
    myLogger.logPrint("Hello world !")
    myLogger.logPrint("Hello world !", "red")
    for i in range(10):
        myLogger.keyValue('minimum', np.random.random())
        myLogger.keyValue('maximum', 15 + np.random.random())
        myLogger.keyValue('average', 20 +np.random.random())
        myLogger.keyValue('serialNumber', 90210)
        myLogger.dumpLogger()
    return status

if __name__ == '__main__':
    status = testLogger()
    print(status)
    