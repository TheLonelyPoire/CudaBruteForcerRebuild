import numpy as np
from getData import *

range_parameters = RP_FINER_EXPANDED_RUN
normalsArr = NS_FINER_EXPANDED_RUN

filteredIndices8 = np.asarray(normalsArr > 7).nonzero()
filteredIndices7 = np.asarray(normalsArr == 7).nonzero()
print("# Normals Pre-Filter:", normalsArr.flatten().shape)
print("# Normals 8-Filter:", filteredIndices8[0].shape)
print("# Normals 7-Filter:", filteredIndices7[0].shape)

# Normals saved as X,Y,Z (or X,Y,ZX_Sum)
with open("yellowNormals.txt", 'w') as file:
    for i in range(filteredIndices8[0].shape[0]):
        file.write(str(range_parameters.minNX + filteredIndices8[1][i] * range_parameters.getXStepSize()) + ',' + 
                   str(range_parameters.minNY + filteredIndices8[0][i] * range_parameters.getYStepSize()) + ',')
        if range_parameters.useZXSum:
            file.write(str(range_parameters.minNZXSum + filteredIndices8[2][i] * range_parameters.getZStepSize()) + '\n')
        else:
            file.write(str(range_parameters.minNZ + filteredIndices8[2][i] * range_parameters.getZStepSize()) + '\n') 

# Indices saved as Y,X,Z (or Y,X,ZX_Sum)
with open("yellowIndices.txt", 'w') as file:
    for i in range(filteredIndices8[0].shape[0]):
        file.write(str(filteredIndices8[0][i]) + ',' + 
                   str(filteredIndices8[1][i]) + ',' + 
                   str(filteredIndices8[2][i]) + '\n')
        
# Normals saved as Y,X,Z (or Y,X,ZX_Sum)
with open("greenNormals.txt", 'w') as file:
    for i in range(filteredIndices7[0].shape[0]):
        file.write(str(range_parameters.minNX + filteredIndices7[1][i] * range_parameters.getXStepSize()) + ',' + 
                   str(range_parameters.minNY + filteredIndices7[0][i] * range_parameters.getYStepSize()) + ',')
        if range_parameters.useZXSum:
            file.write(str(range_parameters.minNZXSum + filteredIndices7[2][i] * range_parameters.getZStepSize()) + '\n')
        else:
            file.write(str(range_parameters.minNZ + filteredIndices7[2][i] * range_parameters.getZStepSize()) + '\n') 

# Indices saved as Y,X,Z (or Y,X,ZX_Sum)
with open("greenIndices.txt", 'w') as file:
    for i in range(filteredIndices7[0].shape[0]):
        file.write(str(filteredIndices7[0][i]) + ',' + 
                   str(filteredIndices7[1][i]) + ',' + 
                   str(filteredIndices7[2][i]) + '\n') 
