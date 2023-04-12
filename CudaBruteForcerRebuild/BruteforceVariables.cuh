#pragma once

#include "cuda_runtime.h"

# define M_PI     3.14159265358979323846  /* pi */
# define MAX_HEIGHT_DIFFERENCE   200.0f;

extern __device__ const short default_triangles[2][3][3];
extern __device__ const float normal_offsets[4][3];


// These two need to be set to constants here for the arrays, so extern isn't an option.

__device__ const int n_y_ranges = 1;
__device__ const int n_floor_ranges = 9;

extern __device__ double lower_y[n_y_ranges];
extern __device__ double upper_y[n_y_ranges];

extern __device__ const double lower_floor[n_floor_ranges];
extern __device__ const double upper_floor[n_floor_ranges];

extern __device__ bool filter_floor_ranges;
extern __device__ bool fall_through_pus;

extern __device__ float magSet[4097];
extern __device__ int magCount;

extern __device__ float gSineTableG[4096];
extern __device__ float gCosineTableG[4096];
extern __device__ int gArctanTableG[8192];
extern __device__ int gReverseArctanTable[65537];

extern __device__ float platform_pos[3];
extern __device__ short startTriangles[2][3][3];
extern __device__ float startNormals[2][3];
extern __device__ bool squishCeilings[4];

extern __device__ float oneUpPlatformNormalYLeft;
extern __device__ float oneUpPlatformNormalXLeft;
extern __device__ float oneUpPlatformYMinLeft;
extern __device__ float oneUpPlatformYMaxLeft;
extern __device__ float oneUpPlatformZMinLeft;
extern __device__ float oneUpPlatformZMaxLeft;
extern __device__ float oneUpPlatformXMinLeft;
extern __device__ float oneUpPlatformXMaxLeft;

extern __device__ float oneUpPlatformNormalYRight;
extern __device__ float oneUpPlatformNormalXRight;
extern __device__ float oneUpPlatformYMinRight;
extern __device__ float oneUpPlatformYMaxRight;
extern __device__ float oneUpPlatformZMinRight;
extern __device__ float oneUpPlatformZMaxRight;
extern __device__ float oneUpPlatformXMinRight;
extern __device__ float oneUpPlatformXMaxRight;

extern __device__ double tenKMultipliers[112];

// Correct Solution Indices (for debugging)
extern __device__ int correctPuSolIdx;
extern __device__ int correctStickSolIdx;
extern __device__ int correctStickSolIdx1;


// Speed Test Passing Variables
extern __device__ int nPass1Sols;
extern __device__ int nPass2Sols;
extern __device__ int nPass3Sols;
extern __device__ int nPass4Sols;
extern __device__ int nPass5Sols;

extern __device__ float currentLowestHeightDiff;


// Common

extern long long int nBlocks;

extern int nPlatSolutionsCPU;
extern int nUpwarpSolutionsCPU;
extern int n10KSolutionsCPU;

// Non-HAU-Aligned

extern int nPUSolutionsCPU;

// HAU-Aligned

extern int nStickSolutionsCPU;
extern int nOUPSolutionsCPU;
extern int nSpeedSolutionsCPU;
