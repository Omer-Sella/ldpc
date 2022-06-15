# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:20:51 2020

@author: Omer
"""
import matplotlib.pyplot as plt
import numpy as np
import copy

COMMON_DATA_TYPE = np.int32
COMMON_INT_DATA_TYPE = np.int32
COMMON_DECIMAL_DATA_TYPE = np.float32
TEST_COORDINATE = 11
COMMON_MATRIX_DATA_TYPE = np.int32
COMMON_PARITY_MATRIX_DIM_1 = 8176
COMMON_PARITY_MATRIX_DIM_0 = 1022

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
def plotDecoderThroughput():
    SNRaxis = ['3.0', '3.2','3.4','3.6']
    throughput50iterations = [1505.682, 2333.213, 4578.976, 7521.208] #
    throughput50iterationsCUDA = [24437.661, 48682.219, 148195.171, 192762.540] #GTX1060Ti
    
    throughput50iterationsCUDARTX3080 = [17069.332, 25092.035, 42335.236, 48736.194] #RTX3080 3.0 3.2 3.4 3.6 200 transmisions iterator frequency check == 6
    ldpcMyImplementation200Iterations = [80.905, 713.747, 3462.618, 6923.815] #Intel Xeon single core
    #   D:\ldpc\ldpc.py:105: NumbaDeprecationWarning: ←[1mThe 'numba.jitclass' decorator has moved to 'numba.experimental.jitclass' to better reflect the experimental nature of the functionality. Please update your imports to accommodate this change and see https://numba.pydata.org/numba-doc/latest/reference/deprecation.html#change-of-jitclass-location for the time frame.←[0m
  # *** In ldpc.py main function.
  # *** in test near earth
  # Time it took the decoder:
  #     5052.834857702255
  #     And the throughput is:
  #         80.90507833970636
  #         Time it took the decoder:
  #             572.751580953598
  #             And the throughput is:
  #                 713.7474842398022
  #                 Time it took the decoder:
  #                     118.06094360351562
  #                     And the throughput is:
  #                         3462.6184369055536
  #                         Time it took the decoder:
  #                             59.04259371757507
  #                             And the throughput is:
  #                                 6923.815067397919
  #                                 ****Time it took for code evaluation == 5807
  #                                 ****Throughput == 281.57104700395854bits per second.

    x = np.arange(len(SNRaxis))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, ldpcMyImplementation200Iterations, width, label='ldpc.py @ maximum 50 iterations')
    rects2 = ax.bar(x + width/2, throughput50iterationsCUDARTX3080, width, label='ldpcCUDA.py @ maximum 50 iterations')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Throughput [bits / second]', size = 16)
    ax.set_xlabel('Signal to noise ratio', size = 16)
    ax.set_title('Decoder throughput for Near-Earth', size = 16)
    ax.set_xticks(x)
    ax.set_xticklabels(SNRaxis)
    ax.legend()

    autolabel(rects1, ax)
    autolabel(rects2, ax)

    
    plt.grid(True, which="both")
    plt.tick_params(axis='both',  labelsize=16)
    
    fig.tight_layout()
    plt.savefig('d:/pythonArt/decoderThroughput.png', format='png', dpi=300)
    plt.show()
    
    return


def plotSNRvsNumberOfIterations(SNRaxis, numberOfIterations, figureNumber = 2, fileName = None):
    plt.figure(figureNumber)
    #plt.clf()
    plt.ylabel('Average Number of iterations',fontsize=16)
    plt.xlabel('Signal to noise ratio (SNR)',fontsize=16)
    plt.title('Current state')
    plt.plot(SNRaxis, numberOfIterations, '^', linewidth = 3)
    plt.tick_params(axis='both',  labelsize=16)
    #fig.set_size_inches(6.25, 6)
    plt.grid(True, which="both")
    plt.tight_layout()
    #plt.show()
    if fileName != None:
        plt.savefig(fileName, format='png', dpi=2000)
    

def plotSNRvsBER(SNRaxis, BERdata, fileName = None, inputLabel = 'baselineCode', figureNumber = 1, figureName = ''):
    snrBaseline = np.array([ 2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. ,  5.5,  6. ,  6.5,  7. ,
         7.5,  8. ,  8.5,  9. ,  9.5, 10. ])
    berPam2 = np.array([3.75061284e-02, 2.96552876e-02, 2.28784076e-02, 1.71725417e-02,
        1.25008180e-02, 8.79381053e-03, 5.95386715e-03, 3.86223164e-03,
        2.38829078e-03, 1.39980484e-03, 7.72674815e-04, 3.98796335e-04,
        1.90907774e-04, 8.39995392e-05, 3.36272284e-05, 1.21088933e-05,
        3.87210822e-06])
    
    snrNearEarthAxis = np.array([3, 3.1, 3.2, 3.3, 3.4])
    snrActualNearEarthAxis = np.array([2.9914482, 3.1541038, 3.3075778, 3.440423]) # 2.9810917, 3.101447, 3.1851501, 3.2885435, 3.4165282
    berNearEarthAxis = np.array([0.02354207, 0.013591, 0.01078767, 0]) # 0.02217123, 0.01199022, 0.00777593, 0.00305577, 0.00045499
    #timeStamp = str(time.time())
    #fileName = "./" + timeStamp + fileName
    #fig, ax = plt.subplots()
    #plt.figure(figureNumber)
    #plt.clf()
    plt.ylabel('Output Bit Error Ratio (BER)',fontsize=16)
    plt.xlabel('Signal to noise ratio (SNR)',fontsize=16)
    plt.title(figureName)
    
    #plt.semilogy(snrBaseline, berPam2, '--b', linewidth = 2, label = 'Uncoded PAM 2')
    plt.plot(snrBaseline, berPam2, '--b', linewidth = 2, label = 'Uncoded PAM 2')
    
    #plt.semilogy(SNRaxis, BERdata, '^', linewidth = 3, label=inputLabel)
    plt.plot(SNRaxis, BERdata, '^', linewidth = 3, label=inputLabel)
    
    #plt.semilogy(snrActualNearEarthAxis, berNearEarthAxis, '*', linewidth = 3, label = 'NearEarthActual')
    #plt.plot(snrActualNearEarthAxis, berNearEarthAxis, '*', linewidth = 3, label = 'NearEarthActual')
    
    plt.tick_params(axis='both',  labelsize=16)
    #fig.set_size_inches(6.25, 6)
    plt.grid(True, which="both")
    plt.tight_layout()
    #plt.show()
    if fileName != None:
        plt.savefig(fileName, format='png', dpi=2000)
    #return timeStamp
    
class berStatistics:
    def __init__(self, codewordSize = 8176):
        self.stats = []
        self.snrAxis = set()
        self.codewordSize = codewordSize
        self.evaluatedPoints = 0
        
    def addEntry(self, snr, sigma, sigmaActual, berUncoded, berDecoded, numberOfDecoderIterations, maximalNumberOfIterations, wasDecoded):
        assert sigmaActual != 0
        # Omer Sella: Obtain snrDbActual by reverse translating from:
        #SNR = 10 ** (SNRdb/10)  and #sigma = np.sqrt(0.5 / SNR)
        snrActual = 1 / (2 * (sigmaActual ** 2))
        snrDbActual = 10 * np.log10(snrActual)
        self.stats.append(list([snr, snrDbActual, sigma, sigmaActual, berUncoded, berDecoded, numberOfDecoderIterations, maximalNumberOfIterations, wasDecoded]))
        self.snrAxis.add(snr)
        return
    
    def getRawStats(self):
        return self.stats
    
    def getStats(self, codewordSize = None):
        # Omer Sella: this function is depricated, but for compatability it is kept here as a wrapper
        _, _, _, snrAxis, averageSnrAxis, berData, averageNumberOfIterations = self.getStatsV2(codewordSize)
        return snrAxis, averageSnrAxis, berData, averageNumberOfIterations
    
    def union(self, rhs):
        # Notice no codeword length safety
        result = berStatistics()
        result.stats = sorted(self.stats + rhs.stats)
        result.snrAxis = self.snrAxis.union(rhs.snrAxis)
        return result
    
    def add(self, rhs):
        # Notice no codeword length safety
        #Union sorts out the stats using sorted
        result = berStatistics()
        result.stats = self.stats + rhs.stats
        result.snrAxis = self.snrAxis.union(rhs.snrAxis)
        return result


    
    def plotStats(self, codewordSize, fileAndPathName = None):
        snrAxis, averageSnrAxis, berData, averageNumberOfIterations = self.getStats(codewordSize)
        if fileAndPathName != None:
            berSnrFigureFileName = fileAndPathName + 'snrVber.png'
            plotSNRvsBER(snrAxis, berData, berSnrFigureFileName)
            berItrFigureFileName = fileAndPathName + 'snrVitr.png'
            plotSNRvsNumberOfIterations(snrAxis, averageNumberOfIterations, berItrFigureFileName)
        else:
            plotSNRvsBER(snrAxis, berData)
            plotSNRvsNumberOfIterations(snrAxis, averageNumberOfIterations)
            
    def getStatsV2(self, codewordSize = None):
        #Omer Sella: this is where we plug in the message size and actually go from error counting to bit error rate.
        if codewordSize == None:
            codewordSize = self.codewordSize
        dataLen = len(self.stats)
        scatterSNR = np.zeros(dataLen, dtype = COMMON_DECIMAL_DATA_TYPE)
        scatterBER = np.zeros(dataLen, dtype = COMMON_DECIMAL_DATA_TYPE)
        scatterITR = np.zeros(dataLen, dtype = COMMON_DECIMAL_DATA_TYPE)
        
        snrAxis = np.array(sorted(list(self.snrAxis)))
        averageSnrAxis = np.zeros(len(snrAxis), dtype = COMMON_DECIMAL_DATA_TYPE)
        averageNumberOfIterations = np.zeros(len(snrAxis), dtype = COMMON_DECIMAL_DATA_TYPE)
        berData = np.zeros(len(snrAxis), dtype = COMMON_DECIMAL_DATA_TYPE)
        count = np.zeros(len(snrAxis), dtype = COMMON_DECIMAL_DATA_TYPE)
        iterator = 0
        for item in self.stats:
            s = item[0]
            actualS = item[1]
            scatterSNR[iterator] = actualS
            errorCount = item[5]
            scatterBER[iterator] = errorCount / codewordSize
            index = np.where(snrAxis == s)
            averageSnrAxis[index] = averageSnrAxis[index] + actualS
            berData[index] = berData[index] + errorCount
            averageNumberOfIterations[index] + averageNumberOfIterations[index] + item[6]
            scatterITR[iterator] = item[6]
            count[index] = count[index] + 1
            iterator = iterator + 1
        # MATLAB syntax would be ./ i.e.: coordinatewise division.
        averageSnrAxis = averageSnrAxis / count 
        averageNumberOfIterations = averageNumberOfIterations / count
        berData = berData / (count * codewordSize)
        return scatterSNR, scatterBER, scatterITR, snrAxis, averageSnrAxis, berData, averageNumberOfIterations
        
            

def updateReward(ax, figure, xCoordinate, yCoordinate, snrData, berData, colour = None):
    if colour == None:
        c = "blue"
    else:
        c = colour
    print("*** size of snrData: " + str(snrData.size))
    print("*** size of berData: " + str(berData.size))
    ax[xCoordinate, yCoordinate].plot(snrData, berData, c)
    # oss22 removed in code review ax[xCoordinate, yCoordinate].set_yscale('log')
    figure.canvas.draw()
    figure.canvas.flush_events()
    return

def updateBerVSnr(ax, figure, xCoordinate, yCoordinate, snrData, berData, colour = None):
    if colour == None:
        c = "blue"
    else:
        c = colour
    ax[xCoordinate, yCoordinate].clear()
    ax[xCoordinate, yCoordinate].scatter(snrData, berData, c = 'blue')
    # oss22 removed in code review  ax[xCoordinate, yCoordinate].set_yscale('log')
    figure.canvas.draw()
    figure.canvas.flush_events()
    return
    
    
def updateCirculantImage(ax, figure, xCoordinate, yCoordinate, circulant):
    colourMap = 'Greys'
    ax[xCoordinate, yCoordinate].matshow(circulant, cmap=colourMap,  interpolation = None)
    figure.canvas.draw()
    figure.canvas.flush_events()
    return

def spawnGraphics(matrix, subMatrixDimX, subMatrixDimY, withReward = True):
    m,n = matrix.shape
    s = m // subMatrixDimX
    t = n // subMatrixDimY
    assert (s * subMatrixDimX ) == m
    assert (t * subMatrixDimY ) == n
    if withReward:
        widths = (t)*[1] + [3]
    else:
        widths = (t)*[1]
    snrActualNearEarthAxis = np.array([2.9914482, 3.1541038, 3.3075778, 3.440423]) # 2.9810917, 3.101447, 3.1851501, 3.2885435, 3.4165282
    berNearEarthAxis = np.array([0.02354207, 0.013591, 0.01078767, 0])
    
    if withReward:
        fig, axs = plt.subplots(s , t + 1, gridspec_kw={'width_ratios': widths })
    else:
        fig, axs = plt.subplots(s , t)#, gridspec_kw={'width_ratios': widths })
    for i in range(s):
        for j in range(t):
            updateCirculantImage(axs, fig, i, j, matrix[ i * subMatrixDimX : (i + 1) * subMatrixDimX - 1 , j * subMatrixDimY : (j + 1) * subMatrixDimY - 1])
            axs[i,j].set_title("Circulant " + str(i) + "," + str(j), fontsize = 18)
            axs[i,j].get_xaxis().set_visible(False)
            axs[i,j].get_yaxis().set_visible(False)
    # oss22 removed in code review  updateReward(axs, fig, 0, 16, snrActualNearEarthAxis, berNearEarthAxis, colour = 'red')
    if withReward:
        axs[0,16].set_title('SNR to BER linear plots ovelayed')
        axs[1,16].set_title('SNR to BER scatter plot')
    return fig, axs

def pieceWiseLinear(X,slope0, bias0, cutoff):
    return np.where(X < cutoff, slope0 * X + bias0, 0.0)

def pieceWiseFit(snrData, berData):
    from scipy.optimize import curve_fit
    optimalParameters, parametersCovariance = curve_fit(pieceWiseLinear, snrData, berData, p0 = [-0.049, 0.16, 3.4])
    return optimalParameters, parametersCovariance

def recursiveLinearFit(xData, yData, numberOfIterations = 10, earlyStopping = False):
    ber = copy.copy(yData)
    snr = copy.copy(xData)
    itr = 0 #Omer Sella: place holder - in the future we may want to return the iteration at which earlyStopping happened
    while itr < numberOfIterations:
        p = np.polyfit(snr, ber, 1)
        trendP = np.poly1d(p)
        ber = ber[trendP(snr) > 0]
        snr = snr[trendP(snr) > 0]
        itr = itr + 1
    return snr, ber, p, trendP, itr
            


def plotEvaluationData(snr, ber, linearFit = True, fillBetween = True):
    
    optimalParameters, _ = pieceWiseFit(snr, ber )
    p = np.polyfit(snr, ber, 1)
    print(p)
    trendP = np.poly1d(p)
    print(trendP)
    slope = p[0]
    bias = p[1]
    fig, ax = plt.subplots()
    ax.scatter(snr, ber)
    ax.plot(snr, trendP(snr), color = 'g')
    ax.plot(snr, pieceWiseLinear(snr, *optimalParameters), color = 'r')
    ax.set_ylabel('Bit error rate', size = 18)
    ax.set_xlabel('Signal to noise ratio', size = 18)
    ax.set_title('Evaluation data', size = 18)
    #ax.set_xticks(ber)
    #ax.set_xticklabels(snr)
    #ax.legend()
    region = np.arange(2.9,3.9,0.1)
    ax.fill_between(region, trendP(region), 0.035, color = '#FFA500', alpha = 0.5)
    plt.show()
    #reward = -1 * p[0]
    #reward =  0.5 * slope * (self.SNRpoints[-1] ** 2)  + bias * self.SNRpoints[-1] - ( 0.5 * slope * (self.SNRpoints[0] ** 2)  + bias * self.SNRpoints[0])
    #reward = -1 * reward
    return fig, ax

def testGraphics():
    matrix = np.random.randint(0,2, (1022,8176))
    subMatrixDim = 511
    fig, ax = spawnGraphics(matrix, subMatrixDim, subMatrixDim)
    snrBaseline = np.array([ 2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. ,  5.5,  6. ,  6.5,  7. ,
         7.5,  8. ,  8.5,  9. ,  9.5, 10. ])
    berPam2 = np.array([3.75061284e-02, 2.96552876e-02, 2.28784076e-02, 1.71725417e-02,
        1.25008180e-02, 8.79381053e-03, 5.95386715e-03, 3.86223164e-03,
        2.38829078e-03, 1.39980484e-03, 7.72674815e-04, 3.98796335e-04,
        1.90907774e-04, 8.39995392e-05, 3.36272284e-05, 1.21088933e-05,
        3.87210822e-06])
    updateBerVSnr(ax, fig, 1, 16, snrBaseline, berPam2)
    fig.show()
    return fig, ax

def uncompress(compressedMatrix):
    from scipy.linalg import circulant
    circulantSize = 511
    paddingLocations = (np.arange(16) + 1) * (circulantSize + 1 ) - 1
    compressionMask = np.ones(np.int32(2 ** np.ceil(np.log2(COMMON_PARITY_MATRIX_DIM_1))), dtype = bool)
    compressionMask[paddingLocations] = False
    firstRow = np.unpackbits(compressedMatrix[0 : len(compressedMatrix) // 2 ])
    secondRow = np.unpackbits(compressedMatrix[len(compressedMatrix) // 2 : ])
    unpaddedFirstRow = firstRow[compressionMask]
    unpaddedSecondRow = secondRow[compressionMask]
    topRows = np.vstack((unpaddedFirstRow, unpaddedSecondRow))
    newMatrix = np.zeros((COMMON_PARITY_MATRIX_DIM_0,COMMON_PARITY_MATRIX_DIM_1), dtype = COMMON_MATRIX_DATA_TYPE)
    # So by now we have row 0 and row 512 of the parity matrix, and now we need to make circulants out of them.
    for j in range(COMMON_PARITY_MATRIX_DIM_0 // circulantSize):
        for i in range(8192 // circulantSize):
            newMatrix[j * circulantSize : (j + 1) * circulantSize, i * circulantSize :  (i + 1) * circulantSize] = circulant(topRows[j, i * circulantSize : circulantSize * (i + 1)])
    return newMatrix

def compressedStringTocompressedByteArray(compressedCodeString):
    compressedCodeString = compressedCodeString.strip('[')
    compressedCodeString = compressedCodeString.strip(']')
    compressedCodeString = compressedCodeString.split()
    compressedCodeByteArray = np.asarray(compressedCodeString)
    compressedCodeByteArray = compressedCodeByteArray.astype(np.uint8)
    return compressedCodeByteArray

def test_uncompress():
    compressedExample = '[128   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 128   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   8   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 128   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 128   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 128   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0 128   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 128   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  64   0   0   0   0   0   0   0   0   0   0   0   0 128   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  64   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 128   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  16   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   4   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  64   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   8   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  32   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  64   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 128   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  32   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  64   0   0   0   0   0   0 128   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   8   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   4   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  16   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  32   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  64   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  16   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   8   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   8   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  32   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  64   0   0   0   0   0   0   0   0   0 128   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   8   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  64   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   4   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  64   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  16   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 128   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  16   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 128   0   0   0  64   0   0   2   0   0   0  32   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  32   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   8   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  16   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   8   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0  16   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 128   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0]'
    compressedMatrix = compressedExample
    compressedMatrix = compressedMatrix.strip('[')
    compressedMatrix = compressedMatrix.strip(']')
    compressedMatrix = compressedMatrix.split()
    compressedMatrix = np.asarray(compressedMatrix)
    compressedMatrix = compressedMatrix.astype(np.uint8)
    parityMatrix = uncompress(compressedMatrix)
    return parityMatrix


#class statistics():
#    self.

def main():
    test_uncompress()
    fig, ax = testGraphics()
    return fig, ax

if __name__ == '__main__':
    main()
