# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:26:10 2021

@author: Omer Sella
"""
import numpy as np

def numToBits(number, numberOfBits):
    assert number < 16
    assert number >= 0
    newNumber = np.zeros(numberOfBits, dtype = int)
    for j in range(numberOfBits - 1, -1, -1):
        newNumber[j] = newNumber[j] + (number % 2)
        number = number >> 1
    return newNumber