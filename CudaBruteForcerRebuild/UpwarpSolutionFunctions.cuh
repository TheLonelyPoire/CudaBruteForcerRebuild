#include "cuda_runtime.h"


__device__ bool try_pu_xz(float* normal, float* position, short(&current_triangles)[2][3][3], float(&triangle_normals)[2][3], double x, double z, int platSolIdx);

__device__ bool try_pu_x(float* normal, float* position, short(&current_triangles)[2][3][3], float(&triangle_normals)[2][3], float(&T_start)[4][4], float(&T_tilt)[4][4], double x, double x1_min, double x1_max, double x2_min, double x2_max, double platform_min_x, double platform_max_x, double platform_min_z, double platform_max_z, double m, double c_min, double c_max, int q_steps, float max_speed, int platSolIdx);

__device__ bool try_pu_z(float* normal, float* position, short(&current_triangles)[2][3][3], float(&triangle_normals)[2][3], float(&T_start)[4][4], float(&T_tilt)[4][4], double z, double z1_min, double z1_max, double z2_min, double z2_max, double platform_min_x, double platform_max_x, double platform_min_z, double platform_max_z, double m, double c_min, double c_max, int q_steps, float max_speed, int platSolIdx);

__device__ void try_normal(float* normal, float* position, int platSolIdx, float max_speed);

__global__ void find_upwarp_solutions(float max_speed);

// Checks to see if an upwarp solution with the correct upwarpPosition was found.
__global__ void check_upwarp_solutions_for_the_right_one();