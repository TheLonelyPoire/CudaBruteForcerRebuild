import numpy as np
from colorMap import *
from getData import *

import matplotlib
from matplotlib import pyplot as plt


def setupPlot(plotArray, sampleIdx, rParams : RangeParameters, nStages, pauseRate, usePara, colmap):
    plotExtents = rParams.getExtents(usePara)
    plt.figure(figsize=(12, 10))
    implot = plt.imshow(plotArray[sampleIdx,:,:].transpose(), 
                   cmap=colmap, interpolation='nearest', origin='upper',
                   extent=plotExtents, vmin=-1 if usePara else 0, vmax=nStages-1)
    plt.colorbar()  

    aspect_ratio = (plotExtents[2] - plotExtents[3]) / (plotExtents[1] - plotExtents[0])

    # Antisymmetric tick counts are used due to horizontal tick orientation
    if aspect_ratio > 4:
        setupTickCounts = (3,20)
    elif aspect_ratio > 1.6:
        setupTickCounts = (5,15)
    elif aspect_ratio > 0.625:
        setupTickCounts = (10,10)
    elif aspect_ratio > 0.25:
        setupTickCounts = (12,5)
    else:
        setupTickCounts = (15,3)

    plt.xlabel("nX")
    stepSizeX = (rParams.maxNX - rParams.minNX) / (setupTickCounts[0] - 1)
    plt.xticks(np.arange(rParams.minNX, rParams.maxNX + stepSizeX/2, stepSizeX))

    if rParams.useZXSum and not usePara:
        plt.ylabel("|nZ| + |nX|")     
        stepSizeZXSum = (rangeParameters.maxNZXSum - rangeParameters.minNZXSum) / (setupTickCounts[1] - 1)
        plt.yticks(np.arange(rParams.maxNZXSum, rParams.minNZXSum - stepSizeZXSum / 2, -stepSizeZXSum))
    else:
        plt.ylabel("nZ")
        minZBound, maxZBound = rParams.computeZBounds()  
        stepSizeZ = (maxZBound - minZBound) / (setupTickCounts[1] - 1)   
        plt.yticks(np.arange(maxZBound, minZBound - stepSizeZ/2, -stepSizeZ))

    plt.title('nY = ' + str(round(sampleIdx * rParams.getYStepSize() + rParams.minNY,5)))

    return implot, setupTickCounts


def update_image_plot(implot, img, pauseRate : float, colmap : clrs.LinearSegmentedColormap, title=''):
    implot.set_array(img)
    implot.set(cmap=colmap)
    plt.title(title)
    plt.pause(pauseRate)
    return implot


folderName = "../output/ImportantSolutions/"
fileName = "normalStagesReached_4_13_23_39.bin"

# folderName = "../output/ElevationRuns/"
# fileName = "platformHWRs_2_8_1_48.bin"

rangeParameters = getRangeParametersFromFile(getCorrespondingRangeParametersFilename(folderName + fileName))
useParallelogram = False

foundHeightDifference = False

if fileName.startswith("norm"):
    plotArr = getIntDataFromBinaryFile(fileName, folderName=folderName, nSamplesY=rangeParameters.nSamplesNY, nSamplesX=rangeParameters.nSamplesNX, nSamplesZ=rangeParameters.nSamplesNZ)
    
    try:
        file, folder = getCorrespondingHeightDiffFilenameAndFolderPath(folderName + fileName)
        heightDiffArr = getFloatDataFromBinaryFile(file, folder, rangeParameters.nSamplesNY, rangeParameters.nSamplesNX, rangeParameters.nSamplesNZ)
        plotArrH = plotArr.astype(float)
        for i in range(rangeParameters.nSamplesNY):
            for j in range(rangeParameters.nSamplesNX):
                for k in range(rangeParameters.nSamplesNZ):
                    if(plotArr[i,j,k] == 8):
                        plotArrH[i,j,k] = 9 - min(0.01 * heightDiffArr[i,j,k], 1.0)
        
        foundHeightDifference = True
    except:
        print("Couldn't Locate Height Difference File at \'" + folder + file + "\' or encountered other error; Skipping")
else:
    plotArr = getFloatDataFromBinaryFile(fileName, folderName=folderName, nSamplesY=rangeParameters.nSamplesNY, nSamplesX=rangeParameters.nSamplesNX, nSamplesZ=rangeParameters.nSamplesNZ)


# Custom Range Parameters
# minNX = 0.19
# maxNX = 0.25
# minNZorZXSum = 0.57
# maxNZorZXSum = 0.64
# minNY = 0.81
# maxNY = 0.9
# nSamplesNX = 241
# nSamplesNZ = 281
# nSamplesNY = 181
# useZXSum = True
# rangeParameters = RangeParameters(minNX, maxNX, minNZorZXSum, maxNZorZXSum, minNY, maxNY, nSamplesNX, nSamplesNZ, nSamplesNY, useZXSum=useZXSum)

# plotArr, plotArrH = getStitchedRunData([ 
#                               "normalStagesReached_3_31_12_23.bin", 
#                               "normalStagesReached_3_31_17_25.bin", 
#                               "normalStagesReached_3_31_16_8.bin", 
#                               "normalStagesReached_3_29_22_41.bin"], folderName, nSamplesNY, nSamplesNX, nSamplesNZ)
# foundHeightDifference = True

# "normalStagesReached_3_29_16_44.bin",
# "normalStagesReached_3_30_12_25.bin",

# Load pre-existing run
# rangeParameters = RP_FINER_EXPANDED_RUN
# plotArr = NS_FINER_EXPANDED_RUN
# foundHeightDifference = False


if(useParallelogram):
    plotArr = getDataAsParallelogram(plotArr, rangeParameters)
    plotArrH = getDataAsParallelogram(plotArrH, rangeParameters)

numStages = 10
pauseRate = 0.01 

if fileName.startswith("norm"):
    colormap = CM_MARBLER if not useParallelogram else CM_MARBLER_PARA
    if foundHeightDifference:
        colormapH = CM_HEIGHT_GRADIENT if not useParallelogram else CM_HEIGHT_GRADIENT_PARA
elif fileName.startswith("plat"):
    colormap = CM_HWR_BANDS 
elif fileName.startswith("minUp"):
    colormap = CM_UPWARP_SPEED
else:
    colormap = CM_DEFAULT

# colormap = CM_MARBLER if not useParallelogram else CM_MARBLER_PARA
# colormapH = CM_HEIGHT_GRADIENT if not useParallelogram else CM_HEIGHT_GRADIENT_PARA

implot, currentTickCounts = setupPlot(plotArr, 0, rangeParameters, numStages, pauseRate, useParallelogram, colormap)
for ny in range(rangeParameters.nSamplesNY):
    update_image_plot(implot, plotArr[ny,:,:].transpose(), pauseRate, colormap, 'nY = ' + str(round(ny * rangeParameters.getYStepSize() + rangeParameters.minNY,5)))

useHeightDiff = False

while(True):
    sample = input("Enter a sample for NY (0 indexed), a valid command (type 'help' for a list), or -1 to quit: ")

    if sample.isdigit():
        sample = int(sample)

        if sample >= rangeParameters.nSamplesNY:
            print("Sample index is too high! Please enter a lower sample index!")
            continue
        
        if not plt.get_fignums():
            implot, _ = setupPlot(plotArrH if useHeightDiff else plotArr, sample, rangeParameters, numStages, pauseRate, useParallelogram, colormapH if useHeightDiff else colormap)

        if useHeightDiff:
            update_image_plot(implot, plotArrH[sample,:,:].transpose(), pauseRate, colormapH, 'nY = ' + str(round(sample * rangeParameters.getYStepSize() + rangeParameters.minNY,5)))
        else:
            update_image_plot(implot, plotArr[sample,:,:].transpose(), pauseRate, colormap, 'nY = ' + str(round(sample * rangeParameters.getYStepSize() + rangeParameters.minNY,5)))
    
        plt.pause(pauseRate)

        continue

    if sample == 'quit' or sample == '-1':
        break

    if sample == 'help':
        print("\nThis program plots per-normal data collected from the BitFS bruteforcer.\n\nValid commands:\n====================")
        print('0,1,...    - Displays a single nY slice of the data based on the index entered. The input cannot exceed the number of nY samples - 1.')
        print('video      - Displays all nY samples in a continuous video, with the frame delay based on the speed of the plotter and the pause rate.')
        print('chpr <f>   - Changes the pause rate to the input (pause rate is stored as a floating-point variable).')
        print('ct <i> <i> - Changes the tick counts for the plot axes; format is <horizontal> <vertical>. Both values must be strictly positive integers.')
        print('thd        - Toggles whether or not the height difference is displayed (if loaded).')
        print('quit       - Quits the application (typing -1 will do the same).')
        print('help       - Prints this menu.')

        continue

    if sample == 'video':
        if not plt.get_fignums():
            implot, _ = setupPlot(plotArrH if useHeightDiff else plotArr, 0, rangeParameters, numStages, pauseRate, useParallelogram, colormapH if useHeightDiff else colormap)

        for ny in range(rangeParameters.nSamplesNY):
            if useHeightDiff:
                update_image_plot(implot, plotArrH[ny,:,:].transpose(), pauseRate, colormapH, 'nY = ' + str(round(ny * rangeParameters.getYStepSize() + rangeParameters.minNY,5)))
            else:
                update_image_plot(implot, plotArr[ny,:,:].transpose(), pauseRate, colormap, 'nY = ' + str(round(ny * rangeParameters.getYStepSize() + rangeParameters.minNY,5))) 
        
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

    if sample == 'thd':
        if foundHeightDifference:
            useHeightDiff = not useHeightDiff
            print("Height difference plotting is now " + ("enabled" if useHeightDiff else "disabled") + ".")
        else:
            print("Height difference file couldn't be loaded; please ensure height difference file is present and named correctly.")

        continue

    if sample.startswith('ct '):
        nums = sample[3:].strip().split()
        if len(nums) != 2:
            print("Format error: Please enter exactly two positive integers separated by whitespace.")
        elif not nums[0].isdigit():
            print("Format error: First entry could not be parsed as an integer.")
        elif not nums[1].isdigit():
            print("Format error: Second entry could not be parsed as an integer.")
        elif int(nums[0]) <= 0:
            print("Value error: First entry was not positive; please enter positive tick counts.")
        elif int(nums[1]) <= 0:
            print("Value error: Second entry was not positive; please enter positive tick counts.")
        else:
            if not plt.get_fignums():
                implot = setupPlot(plotArrH if useHeightDiff else plotArr, 0, rangeParameters, numStages, pauseRate, useParallelogram, colormapH if useHeightDiff else colormap)

            currentTickCounts = (int(nums[0]), int(nums[1]))
            stepSizeX = (rangeParameters.maxNX - rangeParameters.minNX) / (currentTickCounts[0] - 1)
            plt.xticks(np.arange(rangeParameters.minNX, rangeParameters.maxNX + stepSizeX / 2, stepSizeX))

            if rangeParameters.useZXSum and not useParallelogram:
                stepSizeZXSum = (rangeParameters.maxNZXSum - rangeParameters.minNZXSum) / (currentTickCounts[1] - 1)
                plt.yticks(np.arange(rangeParameters.maxNZXSum, rangeParameters.minNZXSum - stepSizeZXSum/2, -stepSizeZXSum))
            else:
                minZBound, maxZBound = rangeParameters.computeZBounds()
                stepSizeZ = (maxZBound - minZBound) / (currentTickCounts[1] - 1)
                plt.yticks(np.arange(maxZBound, minZBound - stepSizeZ/2, -stepSizeZ))
            plt.pause(0.01)

        continue
    
    print("Command not recognized! Please type 'help' for a list of valid commands.")
