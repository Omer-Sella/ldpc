import fileHandler
import copy
import os
import numpy as np
from scipy.linalg import circulant
#import matplotlib.pyplot as plt
import scipy.io
import common
import hashlib
import os

MATRIX_GENERATOR_INT_DATA_TYPE = np.int32

projectDir = os.environ.get('LDPC')
if projectDir == None:
    import pathlib
    projectDir = pathlib.Path(__file__).parent.absolute()
## Omer Sella: added on 01/12/2020, need to make sure this doesn't break anything.
import sys
sys.path.insert(1, projectDir)


nearEarthParity = fileHandler.readMatrixFromFile(projectDir + '/codeMatrices/nearEarthParity.txt', 1022, 8176, 511, True, False, False)
CIRCULANT_SIZE = 511
path = projectDir + '/testMatrices/'
codewordSize = 8176
for i in range(2):
    for j in range(16):
        newMatrix = copy.deepcopy(nearEarthParity)
        newCirculant = np.zeros((511,511), MATRIX_GENERATOR_INT_DATA_TYPE)
        newMatrix[i * CIRCULANT_SIZE : (i + 1) * CIRCULANT_SIZE, j * CIRCULANT_SIZE : (j + 1) * CIRCULANT_SIZE] = newCirculant
        fileName =  "nearEarth_circulant_" + str(i) + "_" + str(j) + "_set_to_0"
        print(fileName)
        print(path)
        fileHandler.saveCodeInstance(newMatrix, CIRCULANT_SIZE, codewordSize = codewordSize, path = path, fileName = fileName)