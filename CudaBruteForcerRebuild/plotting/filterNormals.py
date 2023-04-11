import numpy as np
from getData import *

range_parameters = RP_FINER_EXPANDED_RUN
normalsArr = NS_FINER_EXPANDED_RUN

filteredIndices8 = np.asarray(normalsArr == 8).nonzero()
filteredIndices7 = np.asarray(normalsArr == 7).nonzero()
filteredIndices6 = np.asarray(normalsArr == 6).nonzero()
filteredIndices5 = np.asarray(normalsArr == 5).nonzero()

print("# Normals Pre-Filter:", normalsArr.flatten().shape)
print("# Normals 8-Filter:", filteredIndices8[0].shape)
print("# Normals 7-Filter:", filteredIndices7[0].shape)
print("# Normals 6-Filter:", filteredIndices6[0].shape)
print("# Normals 5-Filter:", filteredIndices5[0].shape)

def saveFilteredNormals(arr, fileName):
    with open(fileName, 'w') as file:
        for i in range(arr[0].shape[0]):
            file.write(str(range_parameters.minNX + arr[1][i] * range_parameters.getXStepSize()) + ',' + 
                       str(range_parameters.minNY + arr[0][i] * range_parameters.getYStepSize()) + ',')
            if range_parameters.useZXSum:
                file.write(str(range_parameters.minNZXSum + arr[2][i] * range_parameters.getZStepSize()) + '\n')
            else:
                file.write(str(range_parameters.minNZ + arr[2][i] * range_parameters.getZStepSize()) + '\n') 

def saveFilteredNormalsIndices(arr, fileName):
    with open(fileName, 'w') as file:
        for i in range(arr[0].shape[0]):
            file.write(str(arr[0][i]) + ',' + 
                    str(arr[1][i]) + ',' + 
                    str(arr[2][i]) + '\n')

# Normals saved as X,Y,Z (or X,Y,ZX_Sum)
saveFilteredNormals(filteredIndices8, "yellowNormals.txt")

# Indices saved as Y,X,Z (or Y,X,ZX_Sum)
saveFilteredNormalsIndices(filteredIndices8, "yellowIndices.txt")
     
# Normals saved as X,Y,Z (or Y,X,ZX_Sum)
saveFilteredNormals(filteredIndices7, "greenNormals.txt")

# Indices saved as Y,X,Z (or Y,X,ZX_Sum)
saveFilteredNormalsIndices(filteredIndices7, "greenIndices.txt")

# Normals saved as X,Y,Z (or Y,X,ZX_Sum)
saveFilteredNormals(filteredIndices6, "crimsonNormals.txt")

# Indices saved as Y,X,Z (or Y,X,ZX_Sum)
saveFilteredNormalsIndices(filteredIndices6, "crimsonIndices.txt")

# Normals saved as X,Y,Z (or Y,X,ZX_Sum)
saveFilteredNormals(filteredIndices5, "skyBlueNormals.txt")

# Indices saved as Y,X,Z (or Y,X,ZX_Sum)
saveFilteredNormalsIndices(filteredIndices5, "skyBlueIndices.txt")
