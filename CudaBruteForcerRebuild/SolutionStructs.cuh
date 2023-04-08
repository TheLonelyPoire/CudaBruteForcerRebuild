#pragma once

#include <cuda_runtime.h>

// =================================================
// MAX_SOLUTION DEFINTIONS
// =================================================

// Common Definitions

# define MAX_UPWARP_SOLUTIONS  8000000  // Set to 8000000 normally
# define MAX_PLAT_SOLUTIONS    50000    // Set to 200000 for non-HAU-aligned, set to 50000 for HAU-aligned

// Non-HAU-Aligned Definitions

# define MAX_10K_SOLUTIONS     50000
# define MAX_PU_SOLUTIONS      50000000

//HAU-Aligned Definitions

# define MAX_10K_SOLUTIONS_HAU 50000
# define MAX_SPEED_SOLUTIONS   200000
# define MAX_OUP_SOLUTIONS     5000000
# define MAX_STICK_SOLUTIONS   20000000 // Set to 7000000 normally



// =================================================
// STRUCT DECLARATIONS
// =================================================

// Structs used in both solvers

// A collection of parameters that define a platform solution.
struct PlatformSolution {
    float returnPosition[3];
    float endPosition[3];
    float endNormal[3];
    short endTriangles[2][3][3];
    float endTriangleNormals[2][3];
    float penultimateFloorNormalY;
    float penultimatePosition[3];
    int nFrames;
};

// A collection of parameters that define a viable upwarp solution.
struct UpwarpSolution {
    int platformSolutionIdx;
    float upwarpPosition[3];
    int pux;
    int puz;
};



// Structs used exclusively in main.cu

// A collection of parameters that define a viable non-HAU aligned 10k solution.
struct TenKSolution {
    int puSolutionIdx;
    int startFloorIdx;
    float startPosition[3];
    float frame1Position[3];
    float frame2Position[3];
    int frame1QSteps;
    int frame2QSteps;
    int frame3QSteps;
    float pre10Kspeed;
    float pre10KVel[2];
    float returnVel[2];
    int stick10K[2];
    int cameraYaw10K;
    float startPositionLimits[2][3];
};

// A collection of parameters that define a viable PU solution.
struct PUSolution {
    int upwarpSolutionIdx;
    float returnSpeed;
    int angle;
    float stickMag;
    int intendedDYaw;
};



// Structs used exclusively in main_hau.cu

// A collection of parameters that define a viable HAU-aligned PU solution.
struct TenKSolutionHAU {
    int speedSolutionIdx;
    float startPosition[3];
    float frame1Position[3];
    float frame2Position[3];
    int frame1QSteps;
    int frame2QSteps;
    float startSpeedX;
    float startSpeedZ;
    float returnSpeed;
    float returnSpeedX;
    float returnSpeedZ;
    int strainMag;
    int strainDYaw;
};

// A collection of parameters that define a viable speed solution.
struct SpeedSolution {
    int oupSolutionIdx;
    float startSpeed;
};

// A collection of parameters that define a viable 1-Up Platform solution.
struct OUPSolution {
    int stickSolutionIdx;
    int pux;
    int puz;
    int angle;
    int cameraYaw;
};

// A collection of parameters that define a viable stick solution.
struct StickSolution {
    int upwarpSolutionIdx;
    int q1q2;
    int q3;
    int floorIdx;
    int xDir;
    int stickY;
    float startSpeed;
};


// =================================================
// STRUCT VARIABLE DECLARATIONS
// =================================================

// Common Struct Variables

extern __device__ struct PlatformSolution platSolutions[MAX_PLAT_SOLUTIONS];
extern __device__ int nPlatSolutions;

extern __device__ struct UpwarpSolution upwarpSolutions[MAX_UPWARP_SOLUTIONS];
extern __device__ int nUpwarpSolutions;


// Non-HAU-Aligned Variables

extern __device__ struct PUSolution puSolutions[MAX_PU_SOLUTIONS];
extern __device__ int nPUSolutions;

extern __device__ struct TenKSolution tenKSolutions[MAX_10K_SOLUTIONS];
extern __device__ int n10KSolutions;

// HAU-Aligned Variables

extern __device__ struct StickSolution stickSolutions[MAX_STICK_SOLUTIONS];
extern __device__ int nStickSolutions;

extern __device__ struct OUPSolution oupSolutions[MAX_OUP_SOLUTIONS];
extern __device__ int nOUPSolutions;

extern __device__ struct SpeedSolution speedSolutions[MAX_SPEED_SOLUTIONS];
extern __device__ int nSpeedSolutions;

extern __device__ struct TenKSolutionHAU tenKSolutionsHAU[MAX_10K_SOLUTIONS_HAU];
extern __device__ int n10KSolutionsHAU;