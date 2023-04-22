#include "cuda_runtime.h"

#include <fstream>


void write_run_parameters(std::ofstream& wfrp, std::string timestamp);

void print_help(std::string &default_output_path, std::string &default_run_params_path);

void process_argument(int& i, char* argv[], std::string& outFileSolutionData, std::string& outFileRunParams);

__global__ void print_success();



__device__ float atomicMinFloat(float* addr, float value); // This was taken from the following StackOverflow post: https://stackoverflow.com/a/51549250

/** 
Checks whether Mario's position coordinates are within the bounds of the map (+/-8192 in all three axes) or within a PU after applying the short cast.
*/
__device__ bool check_inbounds(const float* mario_pos);

__global__ void set_squish_ceilings(float n0, float n1, float n2, float n3);

__global__ void set_platform_pos(float x, float y, float z);

__global__ void set_platform_normal(float nx, float ny, float nz);

__global__ void calculate_10k_multipliers(int minQ1Q2, int maxQ1Q2, int minQ3, int maxQ3);

__global__ void init_reverse_atan();

__global__ void set_start_triangle(short* tris, float* norms);

__device__ int16_t atan2_lookupG(float z, float x);

__device__ int16_t atan2sG(float z, float x);

__device__ float find_closest_mag(float target);

__global__ void init_mag_set();

__global__ void init_camera_angles();

__device__ int atan2b(double z, double x);

__device__ int calculate_camera_yaw(float* currentPosition, float* lakituPosition);

__device__ void platform_logic_gpu(float* platform_normal, float* mario_pos, short(&triangles)[2][3][3], float(&normals)[2][3], float(&mat)[4][4]);
