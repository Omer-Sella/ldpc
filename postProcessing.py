import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from numpy.polynomial import Polynomial


def postMortem():
    filePath = "D:/ldpc/temp/experiments/1623248772/experimet.txt"
    df = pd.read_csv("D:/ldpc/temp/experiments/1623248772/experiment.txt", sep='\t')
    
    keys = df.columns.values
    
    fig, ax = plt.subplots(4,2)
    
    df.Reward.plot(ax=ax[0,0], subplots=True)
    
    dfEpochNumber = df.groupby(["epochNumber"])
    
    dfEpochNumber.Reward.plot(ax=ax[0,1])
    
    dfRewardAvg = dfEpochNumber.Reward.sum() / dfEpochNumber.stepNumber.max()
    
    dfRewardAvg.plot(ax = ax[1,0], subplots = True)
    
    df.boxplot(column = 'Reward', by= 'epochNumber', ax=ax[1,1])
    
    #df.hist(column = 'iAction', by= 'epochNumber', ax=ax[2,0])
    dfEpochNumber.iAction.plot(ax=ax[2,0])
    dfEpochNumber.jAction.plot(ax=ax[2,1])
    #df.hotBitsAction.plot(ax=ax[3,0])
    return

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