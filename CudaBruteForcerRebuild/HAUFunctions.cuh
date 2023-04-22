#pragma once 
#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"

#include "SolutionStructs.cuh"


__global__ void copy_solution_pointers_hau(HAUSolStruct s);

void init_solution_structs_hau(HAUSolStruct* s);

void free_solution_pointers_hau(HAUSolStruct* s);

__global__ void test_speed_solution();

__global__ void test_oup_solution();

__global__ void check_speed_angle();

__global__ void find_stick_solutions();

void run_hau_bruteforcer(int g, int h, int i, int j, float normX, float normY, float normZ, float* host_norms, std::ofstream& wf, char* normalStages, float* finalHeightDiffs);

// Checks to see if a PU solution with the correct parameters was found.
__global__ void check_stick_solutions_for_the_right_one();

// Sets up the output csv column headers.
void setup_output_hau(std::ofstream& wf);