#pragma once

#include <cuda_runtime.h>

// =================================================
// MAX_SOLUTION DEFINTIONS
// =================================================

// Common Definitions

# define MAX_UPWARP_SOLUTIONS  10000000  // Set to 8000000 for Non-HAU/HAU, 10000000 for Slide Kick
# define MAX_PLAT_SOLUTIONS    200000    // Set to 200000 for Non-HAU-aligned or Slide Kick, 50000 for HAU-aligned

// Non-HAU-Aligned Definitions

# define MAX_10K_SOLUTIONS_NON_HAU     50000
# define MAX_PU_SOLUTIONS      50000000

//HAU-Aligned Definitions

# define MAX_10K_SOLUTIONS_HAU 50000
# define MAX_SPEED_SOLUTIONS_HAU   200000
# define MAX_OUP_SOLUTIONS     5000000
# define MAX_STICK_SOLUTIONS   20000000 // Set to 7000000 normally

// Slide Kick Definitions

# define MAX_SK_PHASE_ONE 50000
# define MAX_SK_PHASE_TWO_A 50000
# define MAX_SK_PHASE_TWO_B 50000
# define MAX_SK_PHASE_TWO_C 5000000
# define MAX_SK_PHASE_TWO_D 5000000
# define MAX_SK_PHASE_THREE 4000000
# define MAX_SK_PHASE_FOUR 5000000
# define MAX_SK_PHASE_FIVE 5000000
# define MAX_SK_PHASE_SIX 200000


# define MAX_SK_UPWARP_SOLUTIONS 500000
# define MAX_SPEED_SOLUTIONS_SK   300000000
# define MAX_10K_SOLUTIONS_SK 50000
# define MAX_SLIDE_SOLUTIONS   50000


// =================================================
// STRUCT DECLARATIONS
// =================================================

// Structs used in all solvers

// A collection of parameters that define a platform solution.
struct PlatformSolution {
    float returnPosition[3];
    float returnNormal[3];
    float endPosition[3];
    float endNormal[3];
    short endTriangles[2][3][3];
    float endTriangleNormals[2][3];
    int endFloorIdx;
    float landingFloorNormalsY[3];
    float landingPositions[3][3];
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



// Non-HAU-Aligned Structs

struct NonHAUSolStruct
{
    struct PlatformSolution* platSolutions;
    struct UpwarpSolution* upwarpSolutions;
    struct PUSolution* puSolutions;
    struct TenKSolutionNonHAU* tenKSolutions;
};

// A collection of parameters that define a viable non-HAU aligned 10k solution.
struct TenKSolutionNonHAU {
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



// HAU-Aligned Structs

struct HAUSolStruct
{
    struct PlatformSolution* platSolutions;
    struct UpwarpSolution* upwarpSolutions;
    struct StickSolution* stickSolutions;
    struct OUPSolution* oupSolutions;
    struct SpeedSolutionHAU* speedSolutions;
    struct TenKSolutionHAU* tenKSolutions;
};

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
};

// A collection of parameters that define a viable speed solution.
struct SpeedSolutionHAU {
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



// Slide Kick Structs

struct SKSolStruct {
    struct SKPhase1* sk1Solutions;
    struct SKPhase2* sk2ASolutions;
    struct SKPhase2* sk2BSolutions;
    struct SKPhase2* sk2CSolutions;
    struct SKPhase2* sk2DSolutions;
    struct SKPhase3* sk3Solutions;
    struct SKPhase4* sk4Solutions;
    struct SKPhase5* sk5Solutions;
    struct SKPhase6* sk6Solutions;
    struct PlatformSolution* platSolutions;
    struct UpwarpSolution* upwarpSolutions;
    struct SKUpwarpSolution* skuwSolutions;
    struct SpeedSolutionSK* speedSolutions;
    struct TenKSolutionSK* tenKSolutions;
    struct SlideSolution* slideSolutions;
};

struct SKPhase1 {
    int x1;
    int z1;
    int q2;
    double minSpeed;
    double maxSpeed;
    double minF1Dist;
    double maxF1Dist;
    int minF1AngleIdx;
    int maxF1AngleIdx;
};

struct SKPhase2 {
    int p1Idx;
    int f2Angle;
    int tenKFloorIdx;
    float lower;
    float upper;
    double sinAngle;
    double cosAngle;
};

struct SKPhase3 {
    int p2Idx;
    int p2Type;
    int x2;
    int z2;
};

struct SKPhase4 {
    int p3Idx;
    int cameraYaw;
    double minM1;
    double maxM1;
    double minN1;
    double maxN1;
    float minPre10KSpeed;
    float maxPre10KSpeed;
    float minPost10KSpeed;
    float maxPost10KSpeed;
    double minAngleDiff;
    double maxAngleDiff;
};

struct SKPhase5 {
    int p4Idx;
    int stickX;
    int stickY;
    int f1Angle;
};

struct SKPhase6 {
    int p5Idx;
    float minPre10KSpeed;
    float maxPre10KSpeed;
    float minPost10KSpeed;
    float maxPost10KSpeed;
    double angleDiff;
};

struct SKUpwarpSolution {
    int skIdx;
    int uwIdx;
    float minSpeed;
    float maxSpeed;
};

struct SpeedSolutionSK {
    int skuwSolutionIdx;
    float returnSpeed;
};

struct TenKSolutionSK {
    int speedSolutionIdx;
    float pre10KSpeed;
    float pre10KVel[2];
    float returnVel[2];
    float startPosition[3];
    float frame1Position[3];
    float frame2Position[3];
};

struct SlideSolution {
    int tenKSolutionIdx;
    float preUpwarpPosition[3];
    float upwarpPosition[3];
    int angle;
    float stickMag;
    int intendedDYaw;
    int postSlideAngle;
};

// =================================================
// STRUCT VARIABLE DECLARATIONS
// =================================================

// Common Struct Variables

extern __device__ struct PlatformSolution* platSolutions;
extern __device__ int nPlatSolutions;

extern __device__ struct UpwarpSolution* upwarpSolutions;
extern __device__ int nUpwarpSolutions;


// Non-HAU-Aligned Variables

extern __device__ struct PUSolution* puSolutions;
extern __device__ int nPUSolutions;

extern __device__ struct TenKSolutionNonHAU* tenKSolutionsNonHAU;
extern __device__ int n10KSolutionsNonHAU;

// HAU-Aligned Variables

extern __device__ struct StickSolution* stickSolutions;
extern __device__ int nStickSolutions;

extern __device__ struct OUPSolution* oupSolutions;
extern __device__ int nOUPSolutions;

extern __device__ struct SpeedSolutionHAU* speedSolutionsHAU;
extern __device__ int nSpeedSolutionsHAU;

extern __device__ struct TenKSolutionHAU* tenKSolutionsHAU;
extern __device__ int n10KSolutionsHAU;

// Slide Kick Variables

// Phase Variables

extern __device__ struct SKPhase1* sk1Solutions;
extern __device__ int nSK1Solutions;

extern __device__ struct SKPhase2* sk2ASolutions;
extern __device__ int nSK2ASolutions;

extern __device__ struct SKPhase2* sk2BSolutions;
extern __device__ int nSK2BSolutions;

extern __device__ struct SKPhase2* sk2CSolutions;
extern __device__ int nSK2CSolutions;

extern __device__ struct SKPhase2* sk2DSolutions;
extern __device__ int nSK2DSolutions;

extern __device__ struct SKPhase3* sk3Solutions;
extern __device__ int nSK3Solutions;

extern __device__ struct SKPhase4* sk4Solutions;
extern __device__ int nSK4Solutions;

extern __device__ struct SKPhase5* sk5Solutions;
extern __device__ int nSK5Solutions;

extern __device__ struct SKPhase6* sk6Solutions;
extern __device__ int nSK6Solutions;

// Non-Phase Variables

extern __device__ struct SKUpwarpSolution* skuwSolutions;
extern __device__ int nSKUWSolutions;

extern __device__ struct SpeedSolutionSK* speedSolutionsSK;
extern __device__ int nSpeedSolutionsSK;

extern __device__ struct TenKSolutionSK* tenKSolutionsSK;
extern __device__ int n10KSolutionsSK;

extern __device__ struct SlideSolution* slideSolutions;
extern __device__ int nSlideSolutions;