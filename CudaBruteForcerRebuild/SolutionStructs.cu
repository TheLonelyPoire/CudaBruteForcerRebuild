#include "SolutionStructs.cuh"

// Common Struct Variables

__device__ struct PlatformSolution platSolutions[MAX_PLAT_SOLUTIONS];
__device__ int nPlatSolutions;

__device__ struct UpwarpSolution upwarpSolutions[MAX_UPWARP_SOLUTIONS];
__device__ int nUpwarpSolutions;


// Non-HAU-Aligned Variables

__device__ struct PUSolution puSolutions[MAX_PU_SOLUTIONS];
__device__ int nPUSolutions;

__device__ struct TenKSolution tenKSolutions[MAX_10K_SOLUTIONS];
__device__ int n10KSolutions;

// HAU-Aligned Variables

__device__ struct StickSolution stickSolutions[MAX_STICK_SOLUTIONS];
__device__ int nStickSolutions;

__device__ struct OUPSolution oupSolutions[MAX_OUP_SOLUTIONS];
__device__ int nOUPSolutions;

__device__ struct SpeedSolution speedSolutions[MAX_SPEED_SOLUTIONS];
__device__ int nSpeedSolutions;

__device__ struct TenKSolutionHAU tenKSolutionsHAU[MAX_10K_SOLUTIONS_HAU];
__device__ int n10KSolutionsHAU;