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



// Non-HAU-Aligned Functions

__device__ bool test_stick_position(int solIdx, int x, int y, float endSpeed, float vel1, float xVel1, float zVel1, int angle, int cameraYaw, float* startPosition, float* oneUpPlatformPosition, float oneUpPlatformXMin, float oneUpPlatformXMax, float oneUpPlatformYMin, float oneUpPlatformYMax, float oneUpPlatformZMin, float oneUpPlatformZMax, float oneUpPlatformNormalX, float oneUpPlatformNormalY, int f, float* frame1Position, float* returnPosition, int q1, int q2, int q3);

__device__ bool test_one_up_position(int solIdx, float* startPosition, float* oneUpPlatformPosition, float* returnPosition, float endSpeed, float oneUpPlatformXMin, float oneUpPlatformXMax, float oneUpPlatformYMin, float oneUpPlatformYMax, float oneUpPlatformZMin, float oneUpPlatformZMax, float oneUpPlatformNormalX, float oneUpPlatformNormalY, int f, int q3, int minQ1, int maxQ1, int minQ2, int maxQ2);

__device__ bool find_10k_route(int solIdx, int f, int d, int h, int e, int q3, int minQ1, int maxQ1, int minQ2, int maxQ2);

__global__ void test_pu_solution(int q3, int minQ1, int maxQ1, int minQ2, int maxQ2);

__device__ void try_pu_slide_angle(struct PlatformSolution* sol, int solIdx, int angleIdx, int floorIdx, double s, float xVel1, float zVel1);

__device__ void find_pu_slide_setup(int solIdx);

__global__ void test_upwarp_solution();

__global__ void testEdge(const float x0, const float x1, const float z0, const float z1, float normalX, float normalY, float normalZ, int maxFrames); // THIS ONE SEEMS LIKE IT COULD BE IN COMMON

void run_non_hau_bruteforcer(int g, int h, int i, int j, float normX, float normY, float normZ, float* host_norms, std::unordered_map<uint64_t, PUSolution>& puSolutionLookup, std::ofstream& wf, char* normalStages);


// Checks to see if an upwarp solution with the correct upwarpPosition was found.
__global__ void check_pu_solutions_for_the_right_one();

// Sets up the output csv column headers.
void setup_output_non_hau(std::ofstream& wf);


// HAU-Aligned Functions

__global__ void test_speed_solution();

__global__ void test_oup_solution();

__global__ void check_speed_angle();

__global__ void find_stick_solutions();

void run_hau_bruteforcer(int g, int h, int i, int j, float normX, float normY, float normZ, float* host_norms, std::ofstream& wf, char* normalStages, float* finalHeightDiffs);


// Checks to see if a PU solution with the correct parameters was found.
__global__ void check_stick_solutions_for_the_right_one();

// Sets up the output csv column headers.
void setup_output_hau(std::ofstream& wf);