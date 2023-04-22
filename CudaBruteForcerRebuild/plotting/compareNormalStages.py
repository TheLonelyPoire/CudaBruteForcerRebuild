import numpy as np
from colorMap import *
from getData import *

import matplotlib
from matplotlib import pyplot as plt

folderName1 = "../output/"
fileName1 = "normalStagesReached_4_22_10_26.bin"

folderName2 = "../output/ImportantSolutions/"
fileName2 = "normalStagesReached_4_14_1_4.bin"

rangeParameters1, solverMode1 = getRunParametersFromFile(getCorrespondingRunParametersFilename(folderName1 + fileName1))
rangeParameters2, solverMode2 = getRunParametersFromFile(getCorrespondingRunParametersFilename(folderName2 + fileName2))
useParallelogram = False
tryUseHeightDifference = False

if rangeParameters1 != rangeParameters2:
    print("Range parameters are not the same; cannot compare these runs.")
    quit(0)


normArr1 = getIntDataFromBinaryFile(fileName1, folderName1, nSamplesY=rangeParameters1.nSamplesNY, nSamplesX=rangeParameters1.nSamplesNX, nSamplesZ=rangeParameters1.nSamplesNZ)
normArr2 = getIntDataFromBinaryFile(fileName2, folderName2, nSamplesY=rangeParameters2.nSamplesNY, nSamplesX=rangeParameters2.nSamplesNX, nSamplesZ=rangeParameters2.nSamplesNZ)

differenceNormArrs = normArr2 - normArr1

uniqueArr, countArr = np.unique(differenceNormArrs, return_counts=True)

print("Unique Values:")
print(uniqueArr)
print("\nUnique Counts:")
print(countArr)