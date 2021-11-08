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

import seaborn as sns



REWARD_FOR_NEAR_EARTH_3_0_TO_3_8 = 0.7958451612664468
REWARD_FOR_NEAR_EARTH_3_0_TO_3_4 = 0.3965108116285836

def postMortemHeatMaps(filePath = None, baseline = REWARD_FOR_NEAR_EARTH_3_0_TO_3_4):
    plt.style.use("seaborn")
    if filePath == None:
        filePath = "D:/ldpc/temp/experiments/1625763063/experiment.txt"
        
    df = pd.read_csv(filePath, sep = '\t')
    
    # Get number of unique epochs
    numberOfEpochs = len(np.unique(df.epochNumber))
    
    # Try to get number of interactions per epoch
    epochLength = len(df) % numberOfEpochs
    if epochLength == 0:
        epochLength = len(df) / numberOfEpochs
    else:
        epochLength = 1
        
    
    
    # For getting the infor from the dataframes - https://stackoverflow.com/questions/39250504/count-occurrences-in-dataframe
    # For heatmaps - https://www.askpython.com/python/examples/heatmaps-in-python
    
    #####
    #jAction heat map
    
    # Group data frame by epoch number and then jAction
    gb2 = df.groupby(['epochNumber', 'jAction']) 
    
    # Pad with 0s where jAction type did not occur
    gb3 = gb2.size().unstack(fill_value = 0)
    jActionHeatMapArray = gb3.to_numpy().T
    
    ax1, fig1 = plt.subplots(figsize = jActionHeatMapArray.shape)
    ax1 = sns.heatmap( jActionHeatMapArray / epochLength, linewidth = 1 , annot = True)
    ax1.set_title( "HeatMap of choices of j (column number in the parity matrix)" )
    
    
    #####
    #iAction heat map
    
    # Group data frame by epoch number and then jAction
    gb2 = df.groupby(['epochNumber', 'iAction']) 
    
    # Pad with 0s where jAction type did not occur
    gb3 = gb2.size().unstack(fill_value = 0)
    iActionHeatMapArray = gb3.to_numpy().T
    
    ax2, fig2 = plt.subplots(figsize = iActionHeatMapArray.shape)
    ax2 = sns.heatmap( iActionHeatMapArray / epochLength, linewidth = 1 , annot = True)
    ax2.set_title( "HeatMap of choices of i (row number in the parity matrix)" )
    
    #####
    #kAction heat map
    
    # Group data frame by epoch number and then jAction
    gb2 = df.groupby(['epochNumber', 'kAction']) 
    
    # Pad with 0s where jAction type did not occur
    gb3 = gb2.size().unstack(fill_value = 0)
    kActionHeatMapArray = gb3.to_numpy().T
    
    ax3, fig3 = plt.subplots(figsize = kActionHeatMapArray.shape)
    ax3 = sns.heatmap( kActionHeatMapArray / epochLength, linewidth = 1 , annot = True)
    ax3.set_title( "HeatMap of choices of i (row number in the parity matrix)" )
    
    
    return iActionHeatMapArray, jActionHeatMapArray, kActionHeatMapArray, df
    
    

def postMortem(filePath = None, baseline = None):
    
    if filePath == None:
        filePath = "D:/ldpc/temp/experiments/1623248772/experiment.txt"
        
        filePath = "D:/ldpc/temp/experiments/1623831779/experiment.txt"
        filePath = "D:/ldpc/temp/experiments/1624005673/experiment.txt" # First experiment with env.reset() using random seed
        filePath = "D:/ldpc/temp/experiments/1623831779/experiment.txt" #Omer Sella: This is an experiment with JUST near earth, i.e.: env.reset() sets the initial state to near earth
        #"D:/ldpc/temp/experiments/1624358648/experiment.txt" # smaller region of SNR
        
        filePath = "D:/ldpc/temp/experiments/1625501948/experiment.txt" #seed 61017406 160steps 
        filePath = "D:/ldpc/temp/experiments/1625502010/experiment.txt" #seed 466555 160steps
        
        
        
    df = pd.read_csv(filePath, sep='\t')
    
    keys = df.columns.values
    
    
    fig, ax = plt.subplots(4,6, figsize = (16, 16))
    
    #df.Reward.plot(ax=ax[0,0], subplots=True)
    #xmin = 0
    #xmax = len(df) - 1
    #if baseline != None: 
    #    ax[0,0].hlines(baseline, xmin, xmax)
    
    dfEpochNumber = df.groupby(["epochNumber"])
    
    
    #dfEpochNumber.Reward.plot(ax=ax[0,1])
    #xmin = 0
    #xmax = len(dfEpochNumber.groups)
    #if baseline != None: 
    #    ax[0,1].hlines(baseline, xmin, xmax)
    
### general actor information
    dfRewardAvg = dfEpochNumber.Reward.sum() / dfEpochNumber.stepNumber.max()
    
    dfRewardAvg.plot(ax = ax[0,0], subplots = True)
    xmin = 0
    xmax = len(dfEpochNumber.groups)
    if baseline != None: 
        ax[0,0].hlines(baseline, xmin, xmax)
    ax[0,0].set_title('Averaged undiscounted reward as a function of epoch number')
    ax[0,0].set_ylabel('Reward')
    ax[0,0].set_xlabel('Epoch number')
    
    
    df.logP.plot(ax=ax[1,0])
    ax[1,0].set_title('log probability')
    ax[1,0].set_xlabel('Actor-environment interaction number')
    
    df.actorEntropy.plot(ax=ax[2,0])
    ax[2,0].set_title('Actor entropy')
    ax[2,0].set_xlabel('Actor-environment interaction number')
    
    
    df.boxplot(column = 'Reward', by= 'epochNumber', ax=ax[3,0])
    xmin = 0
    xmax = len(dfEpochNumber.groups)
    if baseline != None: 
        ax[3,0].hlines(baseline, xmin, xmax)
    ax[3,0].set_title('Boxplot of undiscounted reward per epoch number')
    ax[3,0].set_ylabel('Reward')
    ax[3,0].set_xlabel('Epoch number')
    #df.hist(column = 'iAction', by= 'epochNumber', ax=ax[2,0])


### i information
    dfEpochNumber.iAction.plot(ax=ax[0,1])
    if 'logpI' in df:
        dfEpochNumber.logpI.plot(ax=ax[1,1])
        ax[1,1].set_title('log probability')
        ax[1,1].set_ylabel("i")
        ax[1,1].set_xlabel('Actor-environment interaction number')
    ax[0,1].set_title('Choice of i (0 or 1) as a function of actor-environment interaction number')
    ax[0,1].set_ylabel("i [0 or 1]")
    ax[0,1].set_xlabel('Actor-environment interaction number')
    if 'iEntropy' in df:
        dfEpochNumber.iEntropy.plot(ax=ax[2,1])
    
### j information   
    dfEpochNumber.jAction.plot(ax=ax[0,2])
    ax[0,2].set_title('Choice of j (0,1..15) as a function of actor-environment interaction number')
    ax[0,2].set_ylabel("j [0,1..15]")
    ax[0,2].set_xlabel('Actor-environment interaction number')
    if 'logpJ' in df:
        dfEpochNumber.logpJ.plot(ax=ax[1,2])
        ax[1,2].set_title('log probability')
        ax[1,2].set_ylabel("j")
        ax[1,2].set_xlabel('Actor-environment interaction number')
    if 'jEntropy' in df:
        dfEpochNumber.jEntropy.plot(ax=ax[2,2])
    
### k information   
    if 'kAction' in df:
        dfEpochNumber.kAction.plot(ax=ax[0,3])
    if 'logpK' in df:
        dfEpochNumber.logpK.plot(ax=ax[1,3])
    if 'kEntropy' in df:
        dfEpochNumber.kEntropy.plot(ax=ax[2,3])
    
    
    #df.hotBitsAction.plot(ax=ax[3,0])
    
    
    #ax[0,0].set_title('Undiscounted reward as a function of actor-environment interaction number')
    #ax[0,0].set_ylabel('Reward')
    #ax[0,0].set_xlabel('Actor-environment interaction number')
    
    
    #ax[0,1].set_title('Undiscounted reward as a function of actor-environment interaction number, with colour')
    #ax[0,1].set_ylabel('Reward')
    #ax[0,1].set_xlabel('Actor-environment interaction number')
    
 
    
    
    pathBreakdown = os.path.split(filePath)
    imageName = pathBreakdown[0] + "/postProcessing.png"
    plt.tight_layout()
    plt.savefig(fname = imageName)
    
       
    
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