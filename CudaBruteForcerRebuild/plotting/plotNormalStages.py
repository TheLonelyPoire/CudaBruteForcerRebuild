import numpy as np
from colorMap import *
from getData import *

import matplotlib
from matplotlib import pyplot as plt


def setupPlot(plotArray, sampleIdx, rParams : RangeParameters, nStages, pauseRate, usePara):
    implot = plt.imshow(plotArray[sampleIdx,:,:].transpose(), 
                   cmap=colormap, interpolation='nearest', origin='upper',
                   extent=rParams.getExtents(usePara),
                   vmin=-1 if usePara else 0, vmax=nStages-1)
    plt.colorbar()    
    plt.xlabel("nX")
    plt.xticks(np.arange(rParams.minNX, rParams.maxNX + rParams.getXStepSize(), rParams.getXStepSize()*20))

    if rParams.useZXSum and not usePara:
        plt.ylabel("|nZ| + |nX|")     
        plt.yticks(np.arange(rParams.maxNZXSum, rParams.minNZXSum - rParams.getZStepSize(), -rParams.getZStepSize()*10))
    else:
        plt.ylabel("nZ")
        minZBound, maxZBound = rParams.computeZBounds()     
        plt.yticks(np.arange(maxZBound, minZBound - rParams.getZStepSize(), -rParams.getZStepSize()*10))
    
    plt.title('nY = ' + str(round(sampleIdx * rParams.getYStepSize() + rParams.minNY,5)))

    return implot


def update_image_plot(implot, img, pauseRate, title=''):
    implot.set_array(img)
    plt.title(title)
    plt.pause(pauseRate)
    return implot


folderName = "../output/ImportantSolutions/"
fileName = "normalStagesReached_2_9_21_36.bin"

# folderName = "../output/ElevationRuns/"
# fileName = "platformHWRs_2_8_1_48.bin"

rangeParameters = getRangeParametersFromFile(getCorrespondingRangeParametersFilename(folderName + fileName))
useParallelogram = False

# Custom Range Parameters
minNX = -0.213
maxNX = -0.209
minNZorZXSum = 0.598
maxNZorZXSum = 0.602
minNY = 0.8
maxNY = 0.9
nSamplesNX = 201
nSamplesNZ = 201
nSamplesNY = 41
useZXSum = True

# rangeParameters = RangeParameters(minNX, maxNX, minNZorZXSum, maxNZorZXSum, minNY, maxNY, nSamplesNX, nSamplesNZ, nSamplesNY, useZXSum=useZXSum)

if fileName.startswith("norm"):
    plotArr = getIntDataFromBinaryFile(fileName, folderName=folderName, nSamplesY=rangeParameters.nSamplesNY, nSamplesX=rangeParameters.nSamplesNX, nSamplesZ=rangeParameters.nSamplesNZ)
else:
    plotArr = getFloatDataFromBinaryFile(fileName, folderName=folderName, nSamplesY=rangeParameters.nSamplesNY, nSamplesX=rangeParameters.nSamplesNX, nSamplesZ=rangeParameters.nSamplesNZ)

rangeParameters = RP_FINER_EXPANDED_RUN
plotArr = NS_FINER_EXPANDED_RUN

if(useParallelogram):
    plotArr = getDataAsParallelogram(plotArr)

numStages = 10
pauseRate = 0.05 

if fileName.startswith("norm"):
    colormap = CM_EXTRA_STAGES
else:
    colormap = CM_HWR_BANDS

implot = setupPlot(plotArr, 0, rangeParameters, numStages, pauseRate, useParallelogram)
for ny in range(rangeParameters.nSamplesNY):
    update_image_plot(implot, plotArr[ny,:,:].transpose(), pauseRate, 'nY = ' + str(round(ny * rangeParameters.getYStepSize() + rangeParameters.minNY,5)))

while(True):
    sample = input("Enter a sample for NY (0 indexed), a valid command (type 'help' for a list), or -1 to quit: ")

    if sample.isdigit():
        sample = int(sample)

        if sample >= rangeParameters.nSamplesNY:
            print("Sample index is too high! Please enter a lower sample index!\n")
            continue

        update_image_plot(implot, plotArr[sample,:,:].transpose(), pauseRate, 'nY = ' + str(round(sample * rangeParameters.getYStepSize() + rangeParameters.minNY,5)))
    
        plt.pause(pauseRate)

        continue

    if sample == 'quit' or sample == '-1':
        break

    if sample == 'help':
        print("\nThis program plots per-normal data collected from the BitFS bruteforcer.\n\nValid commands:\n====================")
        print('0,1,...  - Displays a single nY slice of the data based on the index entered. The input cannot exceed the number of nY samples - 1.')
        print('video    - Displays all nY samples in a continuous video, with the frame delay based on the speed of the plotter and the pause rate.')
        print('chpr <f> - Changes the pause rate to the input (pause rate is stored as a floating-point variable).')
        print('quit     - Quits the application (typing -1 will do the same).')
        print('help     - Prints this menu.')

        continue

    if sample == 'video':
        for ny in range(rangeParameters.nSamplesNY):
            update_image_plot(implot, plotArr[ny,:,:].transpose(), pauseRate, 'nY = ' + str(round(ny * rangeParameters.getYStepSize() + rangeParameters.minNY,5)))
        continue


    if sample.startswith('chpr '):
        try:
            if(float(sample[5:].strip()) > 0):
                pauseRate = float(sample[5:].strip())
                print("Pause rate changed to:", pauseRate)
            else:
                raise ValueError("Pause value must be positive.")
        except:
            print("Invalid pause rate entered; could not change pause rate.")

        continue
    
    print("Command not recognized! Please type 'help' for a list of valid commands.\n")



    

