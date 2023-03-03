#include "cuda_runtime.h"


__device__ void try_position(float* marioPos, float* normal, int maxFrames);

__global__ void cudaFunc(const float minX, const float deltaX, const float minZ, const float deltaZ, const int width, const int height, float normalX, float normalY, float normalZ, int maxFrames);


// Checks to see if a platform solution with the correct endNormal was found.
__global__ void check_platform_solutions_for_the_right_one();