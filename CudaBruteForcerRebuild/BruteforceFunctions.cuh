# pragma once

#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <unordered_map>

#include "cuda.h"
#include "cuda_runtime.h"

#include "SolutionStructs.cuh"


// Common Bruteforcer Functions

void run_common_bruteforcer(int g, int h, int i, int j, float normX, float normY, float normZ, short* host_tris, float* host_norms, short* dev_tris, float* dev_norms, std::unordered_map<uint64_t, PUSolution>& puSolutionLookup, std::ofstream& wf, char* normalStages, float* finalHeightDiffs);

// Computes the height wiggle room number for the specified normal.
void run_max_elevation_computations(int g, int h, int i, int j, float normX, float normY, float normZ, short* host_tris, float* host_norms, short* dev_tris, float* dev_norms, std::unordered_map<uint64_t, PUSolution>& puSolutionLookup, std::ofstream& wf, float* platformHWRs);

// Computes the minimum upwarp speed for the specified normal.
void run_min_upwarp_speed_computations(int g, int h, int i, int j, float normX, float normY, float normZ, short* host_tris, float* host_norms, short* dev_tris, float* dev_norms, std::unordered_map<uint64_t, PUSolution>& puSolutionLookup, std::ofstream& wf, float* minUpwarpSpeeds);


