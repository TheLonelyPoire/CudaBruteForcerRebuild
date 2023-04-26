#include "SolutionStructs.cuh"

// Common Struct Variables

__device__ struct PlatformSolution* platSolutions;
__device__ int nPlatSolutions;

__device__ struct UpwarpSolution* upwarpSolutions;
__device__ int nUpwarpSolutions;


// Non-HAU-Aligned Variables

__device__ struct PUSolution* puSolutions;
__device__ int nPUSolutions;

__device__ struct TenKSolutionNonHAU* tenKSolutionsNonHAU;
__device__ int n10KSolutionsNonHAU;

// HAU-Aligned Variables

__device__ struct StickSolution* stickSolutions;
__device__ int nStickSolutions;

__device__ struct OUPSolution* oupSolutions;
__device__ int nOUPSolutions;

__device__ struct SpeedSolutionHAU* speedSolutionsHAU;
__device__ int nSpeedSolutionsHAU;

__device__ struct TenKSolutionHAU* tenKSolutionsHAU;
__device__ int n10KSolutionsHAU;


// Slide Kick Variables

// Phase Variables

__device__ struct SKPhase1* sk1Solutions;
__device__ int nSK1Solutions;

__device__ struct SKPhase2* sk2ASolutions;
__device__ int nSK2ASolutions;

__device__ struct SKPhase2* sk2BSolutions;
__device__ int nSK2BSolutions;

__device__ struct SKPhase2* sk2CSolutions;
__device__ int nSK2CSolutions;

__device__ struct SKPhase2* sk2DSolutions;
__device__ int nSK2DSolutions;

__device__ struct SKPhase3* sk3Solutions;
__device__ int nSK3Solutions;

__device__ struct SKPhase4* sk4Solutions;
__device__ int nSK4Solutions;

__device__ struct SKPhase5* sk5Solutions;
__device__ int nSK5Solutions;

__device__ struct SKPhase6* sk6Solutions;
__device__ int nSK6Solutions;

// Non-Phase Variables

__device__ struct SKUpwarpSolution* skuwSolutions;
__device__ int nSKUWSolutions;

__device__ struct SpeedSolutionSK* speedSolutionsSK;
__device__ int nSpeedSolutionsSK;

__device__ struct TenKSolutionSK* tenKSolutionsSK;
__device__ int n10KSolutionsSK;

__device__ struct SlideSolution* slideSolutions;
__device__ int nSlideSolutions;

__device__ struct BDSolution* bdSolutions;
__device__ int nBDSolutions;