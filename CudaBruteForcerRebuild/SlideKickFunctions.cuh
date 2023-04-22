#pragma once

#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"

#include "SolutionStructs.cuh"


void setup_output_slide_kick(std::ofstream& wf);

__global__ void copy_solution_pointers_sk(SKSolStruct s);

void init_solution_structs_sk(SKSolStruct* s);

void free_solution_pointers_sk(SKSolStruct* s);

// Phase Functions

__global__ void try_stick_positionG();

__global__ void try_slide_kick_routeG2();

__global__ void try_slide_kick_routeG(short* pyramidFloorPoints, const int nPoints);

__global__ void find_slide_kick_setupG3a(float platformMinZ, float platformMaxZ);

__global__ void find_slide_kick_setupG3b(float platformMinX, float platformMaxX);

__global__ void find_slide_kick_setupG3c(float platformMinX, float platformMaxX, float platformMinZ, float platformMaxZ);

__global__ void find_slide_kick_setupG3d(float platformMinX, float platformMaxX, float platformMinZ, float platformMaxZ);

__global__ void find_slide_kick_setupG2(short* floorPoints, const int nPoints, float floorNormalY, float platformMinX, float platformMaxX, float platformMinZ, float platformMaxZ, float midPointX, float midPointZ);

__global__ void find_slide_kick_setupG(short* floorPoints, const int nPoints, float floorNormalY, double maxSpeed, int maxF1PU, int t);


// Post-Phase Functions

void find_slide_kick_setup_triangle(short* floorPoints, short* devFloorPoints, int nPoints, float yNormal, int t, double maxSpeed, int nThreads);

// Post-Platform/Upwarp Functions

__device__ float find_pre10K_speed(float post10KSpeed, float& post10KVelX, float& post10KVelZ, int solIdx);

__global__ void test_speed_solution(short* floorPoints, bool* squishEdges, const int nPoints, float floorNormalY);

__global__ void find_speed_solutions();

__global__ void find_sk_upwarp_solutions();

__device__ void try_upwarp_slide(int solIdx, int angle, int intendedDYaw, float intendedMag);

__device__ void try_pu_slide_angle_sk(int solIdx, int angle, double minEndAngle, double maxEndAngle, double minM1, double maxM1);

__global__ void find_slide_solutions();

// Overall Run Bruteforcer Function

void run_slide_kick_bruteforcer(int g, int h, int i, int j, float normX, float normY, float normZ, short* host_tris, float* host_norms, short* dev_tris, float* dev_norms, SKSolStruct s, float normal_offsets_cpu[4][3], short* floorPoints, short* devFloorPoints, bool* squishEdges, bool* devSquishEdges, std::ofstream &wf, char* normalStages);