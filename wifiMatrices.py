import numpy as np
from scipy.linalg import circulant
WIFI_LDPC_DATA_TYPE = np.int64


WIFI_1944_81_5_6 = [[13, 48, 80, 66, 4, 74, 7, 30, 76, 52, 37, 60, None, 49, 73, 31, 74, 73, 23, None, 1, 0, None, None],
[69, 63, 74, 56, 64, 77, 57, 65, 6, 16, 51, None, 64, None, 68, 9, 48, 62, 54, 27, None, 0, 0, None],
[51, 15, 0, 80, 24, 25, 42, 54, 44, 71, 71, 9, 67, 35, None, 58, None, 29, None, 53, 0, None, 0, 0],
[16, 29, 36, 41, 44, 56, 59, 37, 50, 24, None, 65, 4, 65, 52, None, 4, None, 73, 52, 1, None, None, 0]]


def getWifiParityMatrix(codewordSize = 1944, circulantSize = 81, rate = 5/6):
    
    
    if codewordSize == 1944:
        assert circulantSize == 81
        if rate == 5/6:
            for i in range (4):
                for j in range (24):  # 1944 / 81 == 24
                
                    newVector = np.zeros(circulantSize, dtype = WIFI_LDPC_DATA_TYPE)
                    if WIFI_1944_81_5_6[i][j] != None:
                        newVector[WIFI_1944_81_5_6[i][j]]= 1
                    
                    newCirculant = circulant(newVector).T
                    if j != 0:
                        newMatrixHstack = np.hstack((newMatrixHstack, newCirculant))
                    else:
                        newMatrixHstack = newCirculant
                if i != 0:
                    newMatrix = np.vstack((newMatrix, newMatrixHstack))
                else:
                    newMatrix = newMatrixHstack
    return newMatrix
                


