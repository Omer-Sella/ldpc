# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:52:00 2022

@author: omers
"""

from fileHandler import *
import matplotlib.pyplot as plt



def generateNEcompressionExample():
    H = readMatrixFromFile(str(projectDir) + '/codeMatrices/nearEarthParity.txt', 1022, 8176, 511, True, False, False)
    
    for j in range(2):
        for i in range(16):
            fig, ax = plt.subplots()
            colourMap = 'Greys'
            currentCirculant = H[j * 511 : (j + 1) * 511, i * 511 : (i + 1) * 511]
            ax.imshow((currentCirculant) + 1, cmap=colourMap,  interpolation = None)
            ax.set_title("Circulant " + str(j) + "," + str(i), fontsize = 28)
    return

def plotConnectivityMatrix(connectivityMatrix, xLabels = None, yLabels = None, xSize = None, ySize = None, fileName = None, figureSize = None, figParams = None, colour = None, alpha = None, verticalSpacing = None, horizontalSpacing = None):
    
    if xSize is None:
        xFontSize = 28
    else:
        xFontSize = xSize
    
    if ySize is None:
        yFontSize = 28
    else:
        yFontSize = ySize
    
    (verticalDimension, horizontalDimension) = connectivityMatrix.shape
    if figParams is None:
        fig, ax = plt.subplots()
    else:
        fig = figParams[0]
        ax = figParams[1]
        plt.sca(ax)
        
    if colour is None:
        colourMap = 'Greys'
    else:
        colourMap = colour
        
    
    ax.imshow((-1 * connectivityMatrix) + 1, cmap=colourMap,  interpolation = None, alpha = alpha)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    
    if verticalSpacing is None:
        spacingVertical = 5 #verticalDimension // 50
    else:
        spacingVertical = verticalSpacing
    if horizontalSpacing is None:
        spacingHorizontal = 5 #horizontalDimension // 50
    else:
        spacingHorizontal = horizontalSpacing
    
    if xLabels is not None :
        xTickLocations = np.arange(0, horizontalDimension, spacingHorizontal)
        xTickValues = []
        for i in xTickLocations:
            xTickValues.append(xLabels[i])
        plt.xticks(xTickLocations, xTickValues, fontsize = xFontSize, rotation = 90)
    
    if yLabels is not None :
        yTickLocations = np.arange(0, verticalDimension, spacingVertical)
        yTickValues = []
        for i in yTickLocations:
            yTickValues.append(yLabels[i])
    
        plt.yticks(yTickLocations, yTickValues, fontsize = yFontSize)
        
    if fileName is not None:
        if figureSize is not None:
            plt.savefig(fileName, figsize = figureSize)
        else:
            plt.tight_layout()
            plt.savefig(fileName)
            
    return fig, ax