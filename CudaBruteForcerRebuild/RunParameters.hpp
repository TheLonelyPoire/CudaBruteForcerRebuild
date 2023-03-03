#pragma once
#include <string>
#include "vmath.hpp"

// The number of threads to allocate to the GPU kernel.
extern int nThreads;

// The maximum amount of memory available for the program.
extern size_t memorySize;


// Flag for whether to run platform solution max elevation computations.
extern bool computeMaxElevation;

// Flag for whether to run minimum upwarp speed computations (superceded by computeMaxElevation).
extern bool computeMinUpwarp;

// Flag for whether to run the HAU-sligned solver or the non-HAU aligned solver.
extern bool runHAUSolver;

// Flag for whether to parameterize using nZ or by using |nX| + |nZ|
extern bool useZXSum;

// Flag for whether to use the positive value for Z or the negative value of Z when parameterizing by |nX| + |nZ| sum.
extern bool usePositiveZ;

// Flag for whether to stop the bruteforcer after the upwarp stage.
extern bool stopAtUpwarp;

// Flag for whether to print solutions that reach the second-to-last stage in addition to any full solutions (currently only implemented for the HAU-aligned solver).
extern bool printOneOffSolutions;

// The minimum number of quartersteps for the first frame of movement.
extern int minQ1;
// The maximum number of quartersteps for the first frame of movement.
extern int maxQ1;
// The minimum number of quartersteps for the second frame of movement.
extern int minQ2;
// The maximum number of quartersteps for the second frame of movement.
extern int maxQ2;
// The minimum number of quartersteps for the third frame of movement.
extern int minQ3;
// The maximum number of quarter steps for the third frame of movement.
extern int maxQ3;

// The number of frames spent in PUs before setting up for the upwarp.
extern int nPUFrames;
// The maximum number of frames that can be spent on the platform tilting to a valid upwarp position before returning all found platform solutions for the normal (TODO - VERIFY).
extern int maxFrames;

// The minimum value for the platform normal's X component.
extern float minNX;
// The maximum value for the platform normal's X component.
extern float maxNX;
// The minimum value for the platform normal's Z component (only used if 'useZXSum' is set to true).
extern float minNZ;
// The maximum value for the platform normal's Z component (only used if 'useZXSum' is set to false).
extern float maxNZ;
// The minimum value for the platform normal's Y component.
extern float minNY;
// The maximum value for the platform normal's Y component.
extern float maxNY;
// The minimum value for the sum of the magnitudes of the platform normal's X and Z components (only used if 'useZXSum' is set to true).
extern float minNZXSum;
// The maximum value for the sum of the magnitudes of the platform normal's X and Z components (only used if 'useZXSum' is set to true).
extern float maxNZXSum;

// The number of normal X component samples.
extern int nSamplesNX;
// The number of normal Z component samples (this is also used for the number of ZX sum samples in the case where 'useZXSum' is set to true).
extern int nSamplesNZ;
// The number of normal Y component samples.
extern int nSamplesNY;

// Optional file input for centering searches on specific normals (set this to "" for no input)
extern std::string normalsInput;

extern float deltaX;
extern float deltaZ;

// The position of the pyramid platform (TODO - VERIFY).
extern Vec3f platformPos;

// Flag for whether to print additional run information (this is less useful now that a run parameter output file is auto-generated).
extern bool verbose;