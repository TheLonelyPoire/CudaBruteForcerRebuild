import numpy as np

class RangeParameters:
    def __init__(self, minX : float, maxX : float,
                 minZorZXSum : float, maxZorZXSum : float,
                 minY : float, maxY : float,
                 nXs : int, nZs : int, nYs : int, useZXSum=True, useZPositive=True):
        
        self.minNX = minX
        self.maxNX = maxX

        if not useZXSum:
            self.minNZ = minZorZXSum
            self.maxNZ = maxZorZXSum
            self.minNZXSum = None
            self.maxNZXSum = None
            
        else:
            self.minNZ = None
            self.maxNZ = None
            self.minNZXSum = minZorZXSum
            self.maxNZXSum = maxZorZXSum

        self.minNY = minY
        self.maxNY = maxY

        self.nSamplesNX = nXs
        self.nSamplesNZ = nZs

        self.nSamplesNY = nYs

        self.useZXSum = useZXSum
        self.usePositiveZ = useZPositive

    def getXStepSize(self):
        return (self.maxNX - self.minNX) / (self.nSamplesNX - 1)
        
    
    def getZStepSize(self):
        if not self.useZXSum:
            return (self.maxNZ - self.minNZ) / (self.nSamplesNZ - 1)
        else:
            return (self.maxNZXSum - self.minNZXSum) / (self.nSamplesNZ - 1)

    
    def getYStepSize(self):
        return 0 if self.nSamplesNY <= 1 else (self.maxNY - self.minNY) / (self.nSamplesNY - 1) 
    
    def getZXAspectRatio(self):
        return self.nSamplesNZ / self.nSamplesNX

    def getExtents(self, usePara=False):
        minSecond = self.minNZXSum if self.useZXSum else self.minNZ
        maxSecond = self.maxNZXSum if self.useZXSum else self.maxNZ

        if usePara:
            minSecond, maxSecond = self.computeZBounds()

        return [self.minNX - self.getXStepSize()/2, self.maxNX + self.getXStepSize()/2, 
                maxSecond + self.getZStepSize()/2, minSecond - self.getZStepSize()/2]
    
    def evaluateSample(self, idx_y, idx_x, idx_z, eval_z_from_sum=True):
        nY = self.minNY + (self.maxNX - self.minNY) * (idx_y / self.nSamplesNY)
        nX = self.minNX + (self.maxNX - self.minNX) * (idx_x / self.nSamplesNX)
        if not self.useZXSum:
            nZ = self.minNZ + (self.maxNZ - self.minNZ) * (idx_z / self.nSamplesNZ)
        elif eval_z_from_sum:
            nZ = self.minNZXSum + (self.maxNZXSum - self.minNZXSum) * (idx_z / self.nSamplesNZ)
        else:
            nZXSum = self.minNZXSum + (self.maxNZXSum - self.minNZXSum) * (idx_z / self.nSamplesNZ)
            nZ = nZXSum - abs(nX) * (1 if self.usePositiveZ else -1)

        return nY, nX, nZ

    def computeZBounds(self):
        if not self.useZXSum:
            return self.minNZ, self.maxNZ
        else:
            minMagX = min(abs(self.minNX), abs(self.maxNX))
            maxMagX = max(abs(self.minNX), abs(self.maxNX))
            return self.minNZXSum - maxMagX, self.maxNZXSum - minMagX


def getCorrespondingRangeParametersFilename(normStagesPath : str):
    dirIndex = normStagesPath.rfind('/')
    if normStagesPath[dirIndex+1:].startswith("norm"):
        timestampIndex = dirIndex + len("normalStagesReached") + 1
    elif normStagesPath[dirIndex+1:].startswith("plat"):
        timestampIndex = dirIndex + len("platformHWRs") + 1
    elif normStagesPath[dirIndex+1:].startswith("minUp"):
        timestampIndex = dirIndex + len("minUpwarpSpeeds") + 1
    fileExtensionIndex = normStagesPath.rfind('.')

    return normStagesPath[:dirIndex+1] + "runParameters" + normStagesPath[timestampIndex:fileExtensionIndex] + ".txt"


def getRangeParametersFromFile(runParamsPath : str):
    with open(runParamsPath) as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("minNX:"):
                minNX = float(line[7:].strip())
            elif line.startswith("maxNX:"):
                maxNX = float(line[7:].strip())
            elif line.startswith("minNY:"):
                minNY = float(line[7:].strip())
            elif line.startswith("maxNY:"):
                maxNY = float(line[7:].strip())
            elif line.startswith("minNY:"):
                minNZ = float(line[7:].strip())
            elif line.startswith("maxNY:"):
                maxNZ = float(line[7:].strip())
            elif line.startswith("minNZXSum:"):
                minNZXSum = float(line[11:].strip())
            elif line.startswith("maxNZXSum:"):
                maxNZXSum = float(line[11:].strip())
            elif line.startswith("nSamplesNX:"):
                samplesX = int(line[12:].strip())
            elif line.startswith("nSamplesNY:"):
                samplesY = int(line[12:].strip())
            elif line.startswith("nSamplesNZ:"):
                samplesZ = int(line[12:].strip())
            elif line.startswith("Is ZXSum:"):
                isZXSum = bool(line[10:].strip())

    return RangeParameters(minNX, maxNX, 
                    minNZXSum if isZXSum else minNZ, maxNZXSum if isZXSum else maxNZ,
                    minNY, maxNY,
                    samplesX, samplesZ, samplesY, isZXSum)


def getIntDataFromBinaryFile(fileName, folderName="../output/ImportantSolutions/", nSamplesY=0, nSamplesX=0, nSamplesZ=0):
    with open(folderName + fileName, mode='rb') as file: # b is important -> binary
        fileContent = file.read()

        int_values = [i for i in fileContent]

        if nSamplesX == 0 or nSamplesY == 0 or nSamplesZ == 0:
            return np.array(int_values)
        
        return np.array(int_values).reshape((nSamplesY, nSamplesX, nSamplesZ))
    

def getFloatDataFromBinaryFile(fileName, folderName="../output/ImportantSolutions/", nSamplesY=0, nSamplesX=0, nSamplesZ=0):
    with open(folderName + fileName, mode='rb') as file: # b is important -> binary
        if nSamplesX == 0 or nSamplesY == 0 or nSamplesZ == 0:
            return np.fromfile(file, dtype=np.float32)
        
        return np.fromfile(file, dtype=np.float32).reshape((nSamplesY, nSamplesX, nSamplesZ))


def getCorrespondingHeightDiffFilenameAndFolderPath(normStagesPath : str):
    dirIndex = normStagesPath.rfind('/')
    if not normStagesPath[dirIndex+1:].startswith("norm"):
        raise ValueError("MUST INPUT NORMAL STAGES FILE PATH")
    timestampIndex = dirIndex + len("normalStagesReached") + 1
    fileExtensionIndex = normStagesPath.rfind('.')

    return "finalHeightDifferences" + normStagesPath[timestampIndex:fileExtensionIndex] + ".bin", normStagesPath[:dirIndex+1]


def getNormalStagesCoarseRun():
    nSamplesNY, nSamplesNX, nSamplesNZ = 151, 81, 81
    
    folderName = "../output/ImportantSolutions/"

    fileName_75_78 = "normalStagesReached_2_4_16_23.bin"
    fileName_78_80 = "normalStagesReached_2_4_17_32.bin"
    fileName_80_82 = "normalStagesReached_2_4_18_22.bin"
    fileName_82_83 = "normalStagesReached_2_4_19_29.bin"
    fileName_83_87 = "normalStagesReached_2_4_20_31.bin"
    fileName_87_90 = "normalStagesReached_2_4_21_52.bin"

    fileNamesAndCounts = [(fileName_75_78,31),
                            (fileName_78_80,20),
                            (fileName_80_82,20), 
                            (fileName_82_83,10),
                            (fileName_83_87,40),
                            (fileName_87_90,30)]

    normalsArr = np.zeros((nSamplesNY, nSamplesNX, nSamplesNZ))

    counter = 0
    for fileName, yCount in fileNamesAndCounts:
        with open(folderName + fileName, mode='rb') as file: # b is important -> binary
            fileContent = file.read()

        int_values_array = np.array([i for i in fileContent])

        if fileName == fileName_82_83:
            int_values_array = np.reshape(int_values_array, (21, nSamplesNX, nSamplesNZ))
            int_values_array = int_values_array[2::2,:,:]
        elif fileName != fileName_75_78:
            int_values_array = np.reshape(int_values_array, (yCount+1, nSamplesNX, nSamplesNZ))
            int_values_array = int_values_array[1:,:,:]

        normalsArr[counter:counter+yCount,:,:] = np.reshape(int_values_array, (yCount, nSamplesNX, nSamplesNZ))
        counter += yCount

    return normalsArr


def getHWRsCoarseRun():
    nSamplesNY, nSamplesNX, nSamplesNZ = 151, 81, 81

    folderName = "../output/ElevationRuns/"

    fileName = "platformHWRs_2_8_23_54.bin"
    fileName2 = "platformHWRs_2_9_1_4.bin"
    fileName3 = "platformHWRs_2_9_1_53.bin"

    fileNames = [fileName, fileName2, fileName3]

    normalsArr = np.zeros((nSamplesNY, nSamplesNX, nSamplesNZ))

    counter = 0
    for fName in fileNames:
        with open(folderName + fName, mode='rb') as file: # b is important -> binary
            float_values_array = np.fromfile(file, dtype=np.float32)

        numYSamples = 51

        if fName != fileName:
            float_values_array = np.reshape(float_values_array, (numYSamples, nSamplesNX, nSamplesNZ))
            float_values_array = float_values_array[1:,:,:]
            numYSamples = 50

        normalsArr[counter:counter+numYSamples,:,:] = np.reshape(float_values_array, (numYSamples, nSamplesNX, nSamplesNZ))
        counter += numYSamples

    return normalsArr


def getNormalStagesFinerRun():
    nSamplesNY, nSamplesNX, nSamplesNZ = 201, 241, 241
    
    folderName = "../output/ImportantSolutions/"

    fileName_80_81 = "normalStagesReached_2_18_15_19.bin"
    fileName_81_82 = "normalStagesReached_2_19_19_4.bin"
    fileName_82_83 = "normalStagesReached_2_20_19_26.bin"
    fileName_83_84 = "normalStagesReached_2_20_22_1.bin"
    fileName_84_85 = "normalStagesReached_2_21_21_24.bin"
    fileName_85_86 = "normalStagesReached_2_22_22_58.bin"
    fileName_86_87 = "normalStagesReached_2_23_22_58.bin"
    fileName_87_88 = "normalStagesReached_2_24_10_12.bin"
    fileName_88_89 = "normalStagesReached_2_24_19_32.bin"
    fileName_89_90 = "normalStagesReached_2_24_22_36.bin"

    fileNamesAndCounts = [(fileName_80_81,21),
                            (fileName_81_82,20),
                            (fileName_82_83,20), 
                            (fileName_83_84,20),
                            (fileName_84_85,20),
                            (fileName_85_86,20),
                            (fileName_86_87,20),
                            (fileName_87_88,20),
                            (fileName_88_89,20),
                            (fileName_89_90,20)]

    normalsArr = np.zeros((nSamplesNY, nSamplesNX, nSamplesNZ))

    counter = 0
    for fileName, yCount in fileNamesAndCounts:
        with open(folderName + fileName, mode='rb') as file: # b is important -> binary
            fileContent = file.read()

        int_values_array = np.array([i for i in fileContent])

        normalsArr[counter:counter+yCount,:,:] = np.reshape(int_values_array, (yCount, nSamplesNX, nSamplesNZ))
        counter += yCount

    return normalsArr


def getNormalStagesExpandedFinerRun():
    nSamplesNY, nSamplesNX, nSamplesNZ = 221, 241, 281
    
    folderName = "../output/ImportantSolutions/"

    fileName_79_80 = "normalStagesReached_3_22_20_29.bin"

    fileName_80_81 = "normalStagesReached_2_18_15_19.bin"
    fileName_81_82 = "normalStagesReached_2_19_19_4.bin"
    fileName_82_83 = "normalStagesReached_2_20_19_26.bin"
    fileName_83_84 = "normalStagesReached_2_20_22_1.bin"
    fileName_84_85 = "normalStagesReached_2_21_21_24.bin"
    fileName_85_86 = "normalStagesReached_2_22_22_58.bin"
    fileName_86_87 = "normalStagesReached_2_23_22_58.bin"
    fileName_87_88 = "normalStagesReached_2_24_10_12.bin"
    fileName_88_89 = "normalStagesReached_2_24_19_32.bin"
    fileName_89_90 = "normalStagesReached_2_24_22_36.bin"

    fileName_80_90_strip = "normalStagesReached_3_20_20_9.bin"

    ogfileNamesAndCounts = [(fileName_80_81,21),
                            (fileName_81_82,20),
                            (fileName_82_83,20), 
                            (fileName_83_84,20),
                            (fileName_84_85,20),
                            (fileName_85_86,20),
                            (fileName_86_87,20),
                            (fileName_87_88,20),
                            (fileName_88_89,20),
                            (fileName_89_90,20)]

    normalsArr = np.zeros((nSamplesNY, nSamplesNX, nSamplesNZ))

    with open(folderName + fileName_79_80, mode='rb') as file: # b is important -> binary
            fileContent = file.read()
    
    int_values_array = np.array([i for i in fileContent])

    normalsArr[0:21,:,:] = np.reshape(int_values_array, (21, 241, 281))

    counter = 20
    for fileName, yCount in ogfileNamesAndCounts:
        with open(folderName + fileName, mode='rb') as file: # b is important -> binary
            fileContent = file.read()

        int_values_array = np.array([i for i in fileContent])

        normalsArr[counter:counter+yCount,:,:241] = np.reshape(int_values_array, (yCount, nSamplesNX, 241))
        counter += yCount

    with open(folderName + fileName_80_90_strip, mode='rb') as file: # b is important -> binary
            fileContent = file.read()
    
    int_values_array = np.array([i for i in fileContent])

    normalsArr[20:,:,240:] = np.reshape(int_values_array, (201, 241, 41))

    return normalsArr


def getDataAsParallelogram(normsArray : np.ndarray, rangeParams : RangeParameters):
    if normsArray.ndim != 3:
        raise ValueError('Invalid number of array dimensions; should have three dimensions.')
    
    nYs, nXs, nZs = normsArray.shape

    newNormsArray = np.full((nYs, nXs, nZs + nXs - 1), -1)

    for h in range(nYs):
        for i in range(nXs):
            eval_y, eval_x, eval_z = rangeParams.evaluateSample(h,i,0,False)
            if eval_x < 0:
                newNormsArray[h,i,i:i + nZs] = normsArray[h,i,:]
            else:
                newNormsArray[h,i,nXs - 1 - i:nZs + nXs - 1 - i] = normsArray[h,i,:]

    return newNormsArray


def getStitchedRunData(fileNames, folderName, nSamplesY : int, nSamplesX : int, nSamplesZ : int, stitchAxis=0):
    normalsArr = np.zeros((nSamplesY, nSamplesX, nSamplesZ))
    plotHeightsArr = np.full((nSamplesY, nSamplesX, nSamplesZ), 200, dtype=float)

    totalCount = 0
    for fileName in fileNames:
        with open(folderName + fileName, mode='rb') as file: # b is important -> binary
            fileContent = file.read()

        int_values_array = np.array([i for i in fileContent])

        if stitchAxis == 0:
            currentCount = int(int_values_array.size / nSamplesX / nSamplesZ)
            plotArr = normalsArr[totalCount:totalCount+currentCount,:,:] = np.reshape(int_values_array, (currentCount, nSamplesX, nSamplesZ))
        elif stitchAxis == 1:
            currentCount = int_values_array.size / nSamplesY / nSamplesZ
            plotArr = normalsArr[:,totalCount:totalCount+currentCount,:] = np.reshape(int_values_array, (nSamplesY, currentCount, nSamplesZ))
        elif stitchAxis == 2:
            currentCount = int_values_array.size / nSamplesY / nSamplesX
            plotArr = normalsArr[:,:,totalCount:totalCount+currentCount] = np.reshape(int_values_array, (nSamplesY, nSamplesX, currentCount))
        else:
            raise ValueError("Stitch Axis must be between 0 and 2.")

        try:
            file, folder = getCorrespondingHeightDiffFilenameAndFolderPath(folderName + fileName)
            if stitchAxis == 0:
                heightDiffArr = getFloatDataFromBinaryFile(file, folder, currentCount, nSamplesX, nSamplesZ)
            elif stitchAxis == 1:
                heightDiffArr = getFloatDataFromBinaryFile(file, folder, nSamplesY, currentCount, nSamplesZ)
            else:
                heightDiffArr = getFloatDataFromBinaryFile(file, folder, nSamplesY, nSamplesX, currentCount)
            
            plotArrH = plotArr.astype(float)
            for i in range(plotArr.shape[0]):
                for j in range(plotArr.shape[1]):
                    for k in range(plotArr.shape[2]):
                        if(plotArr[i,j,k] == 8):
                            plotArrH[i,j,k] = 9 - min(0.01 * heightDiffArr[i,j,k], 1.0)
            
            if stitchAxis == 0:
                plotHeightsArr[totalCount:totalCount+currentCount,:,:] = plotArrH
            elif stitchAxis == 1:
                plotHeightsArr[:,totalCount:totalCount+currentCount,:] = plotArrH
            elif stitchAxis == 2:
                plotHeightsArr[:,:,totalCount:totalCount+currentCount] = plotArrH

        except:
            print("Couldn't Locate Height Difference File at \'" + folder + file + "\' or encountered other error; Skipping")

        totalCount += currentCount

    return normalsArr, plotHeightsArr


RP_COARSE_RUN = RangeParameters(-0.25, -0.17, 0.56, 0.64, 0.75, 0.9, 81, 81, 151, True)

NS_COARSE_RUN = getNormalStagesCoarseRun()

HWR_COARSE_RUN = getHWRsCoarseRun()

RP_FINER_RUN = RangeParameters(-0.25, -0.19, 0.57, 0.63, 0.8, 0.9, 241, 241, 201, True)

NS_FINER_RUN = getNormalStagesFinerRun()

RP_FINER_EXPANDED_RUN = RangeParameters(-0.25, -0.19, 0.57, 0.64, 0.79, 0.9, 241, 281, 221, True)

NS_FINER_EXPANDED_RUN = getNormalStagesExpandedFinerRun()

# RP_FINER_EXPANDED_RUN_Q2 = RangeParameters(0.19, 0.25, 0.57, 0.64, 0.79, 0.9, 241, 281, 221, True)

# NS_FINER_EXPANDED_RUN_Q2, _ = getStitchedRunData(["normalStagesReached_3_29_16_44.bin",
#                               "normalStagesReached_3_30_12_25.bin", 
#                               "normalStagesReached_3_31_12_23.bin", 
#                               "normalStagesReached_3_31_17_25.bin", 
#                               "normalStagesReached_3_31_16_8.bin", 
#                               "normalStagesReached_3_29_22_41.bin"], "../output/ImportantSolutions/", 221, 241, 281)





