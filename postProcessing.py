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

REWARD_FOR_NEAR_EARTH_3_0_TO_3_8 = 0.7958451612664468
REWARD_FOR_NEAR_EARTH_3_0_TO_3_4 = 0.3965108116285836

def postMortem(filePath = None, baseline = None):
    
    if filePath == None:
        filePath = "D:/ldpc/temp/experiments/1623248772/experiment.txt"
        
        filePath = "D:/ldpc/temp/experiments/1623831779/experiment.txt"
        filePath = "D:/ldpc/temp/experiments/1624005673/experiment.txt" # First experiment with env.reset() using random seed
        filePath = "D:/ldpc/temp/experiments/1623831779/experiment.txt" #Omer Sella: This is an experiment with JUST near earth, i.e.: env.reset() sets the initial state to near earth
        #"D:/ldpc/temp/experiments/1624358648/experiment.txt" # smaller region of SNR
        
        filePath = "D:/ldpc/temp/experiments/1625381750/experiment.txt" #seed 61017406 160steps 
        filePath = "D:/ldpc/temp/experiments/1625295633/experiment.txt" #seed 466555 160steps
        filePath = "D:/ldpc/temp/experiments/1625132296/experiment.txt" #seed 0
        
        
    df = pd.read_csv(filePath, sep='\t')
    
    keys = df.columns.values
    
    
    fig, ax = plt.subplots(4,2)
    
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
    
    
    dfRewardAvg = dfEpochNumber.Reward.sum() / dfEpochNumber.stepNumber.max()
    
    dfRewardAvg.plot(ax = ax[0,0], subplots = True)
    xmin = 0
    xmax = len(dfEpochNumber.groups)
    if baseline != None: 
        ax[0,0].hlines(baseline, xmin, xmax)
    
    df.boxplot(column = 'Reward', by= 'epochNumber', ax=ax[0,1])
    xmin = 0
    xmax = len(dfEpochNumber.groups)
    if baseline != None: 
        ax[0,1].hlines(baseline, xmin, xmax)

    #df.hist(column = 'iAction', by= 'epochNumber', ax=ax[2,0])
    dfEpochNumber.iAction.plot(ax=ax[1,0])
    dfEpochNumber.jAction.plot(ax=ax[1,1])
    if 'kAction' in df:
        dfEpochNumber.kAction.plot(ax=ax[2,1])
    #df.hotBitsAction.plot(ax=ax[3,0])
    
    
    #ax[0,0].set_title('Undiscounted reward as a function of actor-environment interaction number')
    #ax[0,0].set_ylabel('Reward')
    #ax[0,0].set_xlabel('Actor-environment interaction number')
    
    
    #ax[0,1].set_title('Undiscounted reward as a function of actor-environment interaction number, with colour')
    #ax[0,1].set_ylabel('Reward')
    #ax[0,1].set_xlabel('Actor-environment interaction number')
    
    ax[0,0].set_title('Averaged undiscounted reward as a function of epoch number')
    ax[0,0].set_ylabel('Reward')
    ax[0,0].set_xlabel('Epoch number')
    
    ax[0,1].set_title('Boxplot of undiscounted reward per epoch number')
    ax[0,1].set_ylabel('Reward')
    ax[0,1].set_xlabel('Epoch number')
    
    ax[1,0].set_title('Choice of i (0 or 1) as a function of actor-environment interaction number')
    ax[1,0].set_ylabel("i [0 or 1]")
    ax[1,0].set_xlabel('Actor-environment interaction number')
    
    ax[1,1].set_title('Choice of j (0,1..15) as a function of actor-environment interaction number')
    ax[1,1].set_ylabel("j [0,1..15]")
    ax[1,1].set_xlabel('Actor-environment interaction number')
    
    
    
    df.logP.plot(ax=ax[3,0])
    ax[3,0].set_title('log probability')
    ax[3,0].set_xlabel('Actor-environment interaction number')
    
    df.actorEntropy.plot(ax=ax[3,1])
    ax[3,1].set_title('Actor entropy')
    ax[3,1].set_xlabel('Actor-environment interaction number')
    
    pathBreakdown = os.path.split(filePath)
    imageName = pathBreakdown[0] + "/postProcessing.png"
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