import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from numpy.polynomial import Polynomial
import os
import matplotlib.gridspec as gridspec
import ldpcCUDA
import seaborn as sns
import copy
import common
import fileHandler
from plotly.offline import plot
import plotly.graph_objects as go



REWARD_FOR_NEAR_EARTH_3_0_TO_3_8 = 0.7958451612664468
REWARD_FOR_NEAR_EARTH_3_0_TO_3_4 = 0.3965108116285836
MAXIMAL_NUMBER_OF_BEST_CODES = 20
POST_MORTEM_SEED = 42 + 61017406 + 1
POST_MORTEM_SNR_POINTS = [3.0, 3.2,3.4,3.6]
POST_MORTEM_NUMBER_OF_TRANSMISSIONS = 30
POST_MORTEM_NUMBER_OF_ITERATIONS = 50


def postMortemBestCodes(filePath = None, baseline = REWARD_FOR_NEAR_EARTH_3_0_TO_3_4):
    if filePath == None:
        filePath = "D:/ldpc/temp/experiments/1625763063/experiment.txt"    
    df = pd.read_csv(filePath, sep = '\t')
    
    
    # Extract all rows that have the maximum reward (return)
    dfBest = df[df['Reward'] >=max(df.Reward)]
    bestCodes = dfBest.Observation.unique()
    evaluationResults = []
    for i in range(len(bestCodes)):
        print(i)
        compressedMatrix = bestCodes[i]
        compressedMatrix = compressedMatrix.strip('[')
        compressedMatrix = compressedMatrix.strip(']')
        compressedMatrix = compressedMatrix.split()
        compressedMatrix = np.asarray(compressedMatrix)
        compressedMatrix = compressedMatrix.astype(np.uint8)
        parityMatrix = common.uncompress(compressedMatrix)
        print(parityMatrix.shape)
        berStats =  ldpcCUDA.evaluateCodeCuda(POST_MORTEM_SEED, POST_MORTEM_SNR_POINTS, POST_MORTEM_NUMBER_OF_ITERATIONS, parityMatrix, POST_MORTEM_NUMBER_OF_TRANSMISSIONS, G = 'None', cudaDeviceNumber = 0 )
        evaluationResults.append(copy.deepcopy(berStats))
    return evaluationResults
        
        


def postMortemHeatMaps(dataFrame = None, axI = None, axJ = None, axK = None, filePath = None):
    plt.style.use("seaborn")
    if filePath != None:
        df = pd.read_csv(filePath, sep = '\t')
        pathBreakdown = os.path.split(filePath)
    else:
        df = dataFrame

    
    # Get number of unique epochs
    numberOfEpochs = len(np.unique(df.epochNumber))
    
    # Try to get number of interactions per epoch
    epochLength = len(df) % numberOfEpochs
    if epochLength == 0:
        epochLength = len(df) / numberOfEpochs
    else:
        epochLength = 1
        
    
    
    # For getting the info from the dataframes - https://stackoverflow.com/questions/39250504/count-occurrences-in-dataframe
    # For heatmaps - https://www.askpython.com/python/examples/heatmaps-in-python
    
    
    #####
    #iAction heat map
    
    # Group data frame by epoch number and then iAction
    gb2 = df.groupby(['epochNumber', 'iAction']) 
    
    # Pad with 0s where jAction type did not occur
    gb3 = gb2.size().unstack(fill_value = 0)
    iActionHeatMapArray = gb3.to_numpy().T
    
    #if axI == None:
    #    axI, fig2 = plt.subplots(figsize = iActionHeatMapArray.shape)
            
    #axI = sns.heatmap( iActionHeatMapArray / epochLength, linewidth = 1 , annot = True)
    #axI.set_title( "HeatMap of choices of i (row number in the parity matrix)" )
    
    
    #if filePath != None:
    #    imageName = pathBreakdown[0] + "/heatMapI.png"
    #    plt.tight_layout()
    #    plt.savefig(fname = imageName)
    
    
    #####
    #jAction heat map
    
    # Group data frame by epoch number and then jAction
    gb2 = df.groupby(['epochNumber', 'jAction']) 
    
    # Pad with 0s where jAction type did not occur
    gb3 = gb2.size().unstack(fill_value = 0)
    jActionHeatMapArray = gb3.to_numpy().T
    
    #if axJ == None:
    #    axJ, fig1 = plt.subplots(figsize = jActionHeatMapArray.shape)
    
    #axJ = sns.heatmap( jActionHeatMapArray / epochLength, linewidth = 1 , annot = True)
    #axJ.set_title( "HeatMap of choices of j (column number in the parity matrix)" )
    
    #f filePath != None:
    #   imageName = pathBreakdown[0] + "/heatMapJ.png"
    #    plt.tight_layout()
    #    plt.savefig(fname = imageName)
    
    
    
    #####
    #kAction heat map
    
    # Group data frame by epoch number and then kAction
    gb2 = df.groupby(['epochNumber', 'kAction']) 
    
    # Pad with 0s where jAction type did not occur
    gb3 = gb2.size().unstack(fill_value = 0)
    kActionHeatMapArray = gb3.to_numpy().T
    
    #if axJ == None:
    #    axJ, fig3 = plt.subplots(figsize = kActionHeatMapArray.shape)
    #axJ = sns.heatmap( kActionHeatMapArray / epochLength, linewidth = 1 , annot = True)
    #axJ.set_title( "HeatMap of choices of k (number of hot bits)" )
    
    #if filePath != None:
    #    imageName = pathBreakdown[0] + "/heatMapK.png"
    #    plt.tight_layout()
    #    plt.savefig(fname = imageName)
    
    
    return iActionHeatMapArray, jActionHeatMapArray, kActionHeatMapArray
    
def seriesGBtoArray(seriesGroupByOBJ):
    countV      = np.array(seriesGroupByOBJ.count())
    numberOfEpochs = len(countV)
    minV        = np.array(seriesGroupByOBJ.min())
    maxV        = np.array(seriesGroupByOBJ.max())
    avgV        = np.array(seriesGroupByOBJ.mean())
    medianV     = np.array(seriesGroupByOBJ.median())
    rawData = np.zeros((numberOfEpochs, np.max(countV)), dtype = np.float64)
#    for i in range(numberOfEpochs):
        #rawData[i,:] = np.array(seriesGroupByOBJ.get_group(i)).T
    
    return countV, numberOfEpochs, minV, maxV, avgV, medianV, rawData
    
    

def postMortem(filePath = None, baseline = None):
    plt.style.use("seaborn")
    if filePath == None:
        filePath = "D:/ldpc/temp/experiments/1623248772/experiment.txt"
        
        filePath = "D:/ldpc/temp/experiments/1623831779/experiment.txt"
        filePath = "D:/ldpc/temp/experiments/1624005673/experiment.txt" # First experiment with env.reset() using random seed
        filePath = "D:/ldpc/temp/experiments/1623831779/experiment.txt" #Omer Sella: This is an experiment with JUST near earth, i.e.: env.reset() sets the initial state to near earth
        #"D:/ldpc/temp/experiments/1624358648/experiment.txt" # smaller region of SNR
        
        filePath = "D:/ldpc/temp/experiments/1625501948/experiment.txt" #seed 61017406 160steps 
        filePath = "D:/ldpc/temp/experiments/1625502010/experiment.txt" #seed 466555 160steps
        
        
        
    df = pd.read_csv(filePath, sep='\t')
    
     # Get number of unique epochs
    numberOfEpochs = len(np.unique(df.epochNumber))
    
    # Try to get number of interactions per epoch
    epochLength = len(df) % numberOfEpochs
    if epochLength == 0:
        epochLength = len(df) / numberOfEpochs
    else:
        epochLength = 1
    
    
    fig, ax = plt.subplots(4,4, figsize = (16, 16))
    figPolicy, axPolicy = plt.subplots(3,3)
    figPerformance, axPerformance = plt.subplots(1,2)
    
    
    #df.Reward.plot(ax=ax[0,0], subplots=True)
    xmin = 0
    xmax = len(df) - 1
    if baseline != None: 
        ax[0,0].hlines(baseline, xmin, xmax)
    
    iActionHeatMapArray, jActionHeatMapArray, kActionHeatMapArray = postMortemHeatMaps(dataFrame = df)#, axI = ax[0,1], axJ = ax[0,2], axK = ax[0,3])
    
    dfEpochNumber = df.groupby(["epochNumber"])
    rewardGroupByEpochSeries = dfEpochNumber.Reward
    countV, numberOfEpochs, minV, maxV, avgV, medianV, rawData = seriesGBtoArray(rewardGroupByEpochSeries)
    xRange = np.arange(numberOfEpochs)
    #xRangeForScatter = np.tile(A, reps)
    #xmin = 0
    #xmax = len(dfEpochNumber.groups)
    #if baseline != None: 
    #    ax[0,1].hlines(baseline, xmin, xmax)
    
### general actor information
    #dfRewardAvg = dfEpochNumber.Reward.mean() #sum() / dfEpochNumber.stepNumber.max()
    
    #if 'vValue' in df.keys():
    #    figV, axV = plt.subplots(1,2)
    #    rewardGroupByEpochSeries = dfEpochNumber.vValue
    #    countV, numberOfEpochs, minV, maxV, avgV, medianV, rawData = seriesGBtoArray(rewardGroupByEpochSeries)
    #    xRange = np.arange(numberOfEpochs)
    #    axV[0].plot(xRange, avgV)
    #    axV[0].plot(xRange, minV, 'r-')
    #    axV[0].plot(xRange, maxV, color = 'green', marker = '+')
    #    axV[0].plot(xRange, medianV, color = 'blue', marker = 'd')
    #    #axPerformance[1].scatter(xRange, rawData)
    #   #dfEpochNumber.Reward.plot(ax = axPerformance[1])
    #    axV.scatter(np.arange(len(np.array(df.vValue))), np.array(df.vValue))
    #    pathBreakdown = os.path.split(filePath)
    #    imageName = pathBreakdown[0] + "/vValue.png"
    #    figV.tight_layout()
    #    figV.show()
    #    figV.savefig(fname = imageName)
        
    
    ax[0,0].plot(xRange, avgV)#dfRewardAvg.plot(by = 'epochNumber', ax = ax[0,0], subplots = True)
    axPerformance[0].plot(xRange, avgV)
    axPerformance[0].plot(xRange, minV, 'r-')
    axPerformance[0].plot(xRange, maxV, color = 'green', marker = '+')
    axPerformance[0].plot(xRange, medianV, color = 'blue', marker = 'd')
    #axPerformance[1].scatter(xRange, rawData)
    #dfEpochNumber.Reward.plot(ax = axPerformance[1])
    axPerformance[1].scatter(np.arange(len(np.array(df.Reward))), np.array(df.Reward))
    xmin = 0
    xmax = numberOfEpochs
    if baseline != None: 
        
        ax[0,0].hlines(baseline, xmin, xmax)
    ax[0,0].set_title('Averaged undiscounted reward as a function of epoch number')
    ax[0,0].set_ylabel('Reward')
    ax[0,0].set_xlabel('Epoch number')
    axPerformance[0].set_title('Averaged undiscounted reward as a function of epoch number')
    axPerformance[0].set_ylabel('Reward')
    axPerformance[0].set_xlabel('Epoch number')
    
    df.logP.plot(ax=ax[1,0])
    ax[1,0].set_title('log probability')
    ax[1,0].set_xlabel('Actor-environment interaction number')
    
    df.actorEntropy.plot(ax=ax[2,0])
    ax[2,0].set_title('Actor entropy')
    ax[2,0].set_xlabel('Actor-environment interaction number')
    
    
    df.boxplot(column = 'Reward', by= 'epochNumber', ax=ax[3,0])
    #df.boxplot(column = 'Reward', by= 'epochNumber', ax=axPerformance[0])
    xmin = 0
    xmax = len(dfEpochNumber.groups)
    if baseline != None: 
        ax[3,0].hlines(baseline, xmin, xmax)
    ax[3,0].set_title('Boxplot of undiscounted reward per epoch number')
    ax[3,0].set_ylabel('Reward')
    ax[3,0].set_xlabel('Epoch number')
    #axPerformance[1].set_title('Boxplot of undiscounted reward per epoch number')
    #axPerformance[1].set_ylabel('Reward')
    #axPerformance[1].set_xlabel('Epoch number')
    #df.hist(column = 'iAction', by= 'epochNumber', ax=ax[2,0])


### i information
    sns.heatmap( iActionHeatMapArray / epochLength, linewidth = 1 , annot = False, ax = ax[0,1])
    sns.heatmap( iActionHeatMapArray / epochLength, linewidth = 1 , annot = False, ax = axPolicy[0,0])
    if 'logpI' in df:
        dfEpochNumber.logpI.plot(ax=ax[1,1])
        ax[1,1].set_title('log probability')
        ax[1,1].set_ylabel("i")
        ax[1,1].set_xlabel('Actor-environment interaction number')
        
        dfEpochNumber.logpI.plot(ax=axPolicy[1,0])
        axPolicy[1,0].set_title('log probability')
        axPolicy[1,0].set_ylabel("i")
        axPolicy[1,0].set_xlabel('Actor-environment interaction number')
    
    
    #dfEpochNumber.iAction.plot(ax=ax[0,1])
    
    #ax[0,1].set_title('Choice of i (0 or 1) as a function of actor-environment interaction number')
    #ax[0,1].set_ylabel("i [0 or 1]")
    #ax[0,1].set_xlabel('Actor-environment interaction number')
    if 'iEntropy' in df:
        dfEpochNumber.iEntropy.plot(ax=ax[2,1])
        dfEpochNumber.iEntropy.plot(ax=axPolicy[2,0])
    
### j information   
    sns.heatmap( jActionHeatMapArray / epochLength, linewidth = 1 , annot = False, ax = ax[0,2])    
    sns.heatmap( jActionHeatMapArray / epochLength, linewidth = 1 , annot = False, ax = axPolicy[0,1])    
    #dfEpochNumber.jAction.plot(ax=ax[0,2])
    #ax[0,2].set_title('Choice of j (0,1..15) as a function of actor-environment interaction number')
    #ax[0,2].set_ylabel("j [0,1..15]")
    #ax[0,2].set_xlabel('Actor-environment interaction number')
    if 'logpJ' in df:
        dfEpochNumber.logpJ.plot(ax=ax[1,2])
        
        ax[1,2].set_title('log probability')
        ax[1,2].set_ylabel("j")
        ax[1,2].set_xlabel('Actor-environment interaction number')
        
        dfEpochNumber.logpJ.plot(ax=axPolicy[1,1])
        axPolicy[1,1].set_title('log probability')
        axPolicy[1,1].set_ylabel("j")
        axPolicy[1,1].set_xlabel('Actor-environment interaction number')
    if 'jEntropy' in df:
        dfEpochNumber.jEntropy.plot(ax=ax[2,2])
        dfEpochNumber.jEntropy.plot(ax=axPolicy[2,1])
    
### k information   
    #if 'kAction' in df:
    #    dfEpochNumber.kAction.plot(ax=ax[0,3])
    sns.heatmap( kActionHeatMapArray / epochLength, linewidth = 1 , annot = False, ax = ax[0,3])
    sns.heatmap( kActionHeatMapArray / epochLength, linewidth = 1 , annot = False, ax = axPolicy[0,2])
    if 'logpK' in df:
        dfEpochNumber.logpK.plot(ax=ax[1,3])
        dfEpochNumber.logpK.plot(ax=axPolicy[1,2])
    if 'kEntropy' in df:
        dfEpochNumber.kEntropy.plot(ax=ax[2,3])
        dfEpochNumber.kEntropy.plot(ax=axPolicy[2,2])
    
    
    #df.hotBitsAction.plot(ax=ax[3,0])
    
    
    #ax[0,0].set_title('Undiscounted reward as a function of actor-environment interaction number')
    #ax[0,0].set_ylabel('Reward')
    #ax[0,0].set_xlabel('Actor-environment interaction number')
    
    
    #ax[0,1].set_title('Undiscounted reward as a function of actor-environment interaction number, with colour')
    #ax[0,1].set_ylabel('Reward')
    #ax[0,1].set_xlabel('Actor-environment interaction number')
    
 
    
    ### Save all figures
    pathBreakdown = os.path.split(filePath)
    imageName = pathBreakdown[0] + "/postProcessing.png"
    fig.tight_layout()
    fig.savefig(fname = imageName)
    
    imageName = pathBreakdown[0] + "/policy.png"
    figPolicy.tight_layout()
    figPolicy.show()
    figPolicy.savefig(fname = imageName)
    
    imageName = pathBreakdown[0] + "/performance.png"
    figPerformance.tight_layout()
    figPerformance.show()
    figPerformance.savefig(fname = imageName)
    
    
    
    ########################################### New flow #########################################
    
       
    
    return df

def drawRewardSurface():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    START_POINT = 2.8
    END_POINT = 3.8
    bias = np.arange(-3,3,0.1)
    slope = np.arange(-3,3,0.1)
    slope, bias = np.meshgrid(slope, bias)
    reward = 0.5 * slope * (END_POINT ** 2)  + bias * END_POINT - ( 0.5 * slope * (START_POINT ** 2)  + bias * START_POINT)
    reward2 = 0.5 * slope * (END_POINT ** 2)  + bias * END_POINT - ( 0.5 * slope * (START_POINT ** 2)  + bias * START_POINT) + END_POINT - START_POINT
    p = np.poly1d([slope, bias])
    pConst = np.poly1d([1])
    pTotalInteg = (pConst - p).integ()
    reward3 = pTotalInteg(END_POINT) - pTotalInteg(START_POINT)
    surf = ax.plot_surface(slope, bias, reward3, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    
    # Customize the z axis.
    ax.set_zlim(-10.01, 10.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    return slope, bias. reward

def drawPoly():
    example = [-0.03318342,  0.11585322] #Taken from running ldpcCUDA.py as standalone
    x = np.linspace(0.0, 5.0, 100)
    p = np.poly1d(example)
    print(p)
    print(p.coefficients)
    p.integ()
    print(p.integ().coefficients)
    return

def extractCodes(dataFrame, path = None, best = 0.395, worst = 0.3855, experimentDir = "D:/ldpc/codeMatrices/experimental/"):
    # extract some codes from a dataFrame (trace)
    # We extract the overall best, overall worst, best in every epoch, worst in every epoch, median in every epoch.
    bestCodes = []
    mask = dataFrame.Reward > best
    pathBest = experimentDir + "/bestCodes/"
    if not os.path.exists(pathBest):
        os.mkdir(pathBest)
    for compressedCode in dataFrame.Observation[mask] :
        compressedCode = common.compressedStringTocompressedByteArray(compressedCode)
        uncompressedCode = common.uncompress(compressedCode)
        fileHandler.saveCodeInstance(uncompressedCode, circulantSize = 511, codewordSize = 8176, evaluationData = None, path = pathBest)
        bestCodes.append(copy.deepcopy(uncompressedCode))
    worstCodes = []
    mask = dataFrame.Reward < worst
    pathWorst = experimentDir + "/worstCodes/"
    if not os.path.exists(pathWorst):
        os.mkdir(pathWorst)
    for compressedCode in dataFrame.Observation[mask] :
        compressedCode = common.compressedStringTocompressedByteArray(compressedCode)
        uncompressedCode = common.uncompress(compressedCode)
        fileHandler.saveCodeInstance(uncompressedCode, circulantSize = 511, codewordSize = 8176, evaluationData = None, path = pathWorst)
        worstCodes.append(copy.deepcopy(uncompressedCode))
    
    dfGroupByEpoch = dataFrame.groupby(['epochNumber'])
    
    epochWorstCodes = dataFrame.loc[dfGroupByEpoch.Reward.idxmin()].Observation
    worstPerEpoch = []
    pathEpochWorst = experimentDir + "/worstPerEpoch/"
    if not os.path.exists(pathEpochWorst):
        os.mkdir(pathEpochWorst)
    for compressedCode in epochWorstCodes:
        compressedCode = common.compressedStringTocompressedByteArray(compressedCode)
        uncompressedCode = common.uncompress(compressedCode)
        fileHandler.saveCodeInstance(uncompressedCode, circulantSize = 511, codewordSize = 8176, evaluationData = None, path = pathEpochWorst)
        worstPerEpoch.append(copy.deepcopy(uncompressedCode))
        
    epochBestCodes = dataFrame.loc[dfGroupByEpoch.Reward.idxmax()].Observation
    bestPerEpoch = []
    pathEpochBest = experimentDir + "/bestPerEpoch/"
    if not os.path.exists(pathEpochBest):
        os.mkdir(pathEpochBest)
    for compressedCode in epochBestCodes:
        compressedCode = common.compressedStringTocompressedByteArray(compressedCode)
        uncompressedCode = common.uncompress(compressedCode)
        fileHandler.saveCodeInstance(uncompressedCode, circulantSize = 511, codewordSize = 8176, evaluationData = None, path = pathEpochBest)
        bestPerEpoch.append(copy.deepcopy(uncompressedCode))
    
    return bestCodes, worstCodes, worstPerEpoch, bestPerEpoch
    
    
def crawler(pathToExperiments):
    return
    

def analysisOfTwoPopulations(dataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(dataFrame.Reward.count())), y=list(dataFrame.Reward) ))
    # Set title
    fig.update_layout(title_text="Reward as a function of time")
    # style all the traces
    fig.update_traces(hoverinfo="name+x+text", line={"width": 0.5}, marker={"size": 8}, mode="lines+markers", showlegend=False)
    plot(fig, auto_open=True)
    
    return



    