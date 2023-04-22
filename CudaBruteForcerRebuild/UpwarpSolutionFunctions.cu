#include "UpwarpSolutionFunctions.cuh"

#include "math.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#include "Platform.cuh"

#include "BruteforceVariables.cuh"
#include "CommonFunctions.cuh"
#include "RunParameters.hpp"
#include "SolutionStructs.cuh"


__device__ bool try_pu_xz(float* normal, float* position, short(&current_triangles)[2][3][3], float(&triangle_normals)[2][3], double x, double z, int platSolIdx) {
    // For current (x, z) PU position, find range of yaws that
    // allow you to reach the PU platform from the original universe.

    float test_normal[3] = { normal[0], normal[1], normal[2] };
    float mario_pos[3] = { x + position[0], position[1], z + position[2] };

    short triangles[2][3][3];
    float normals[2][3];
    float mat[4][4];

    platform_logic_gpu(test_normal, mario_pos, triangles, normals, mat);

    if (check_inbounds(mario_pos)) {
        for (int i = 0; i < n_y_ranges; i++) {
            if (mario_pos[1] >= *(lower_y + i) && mario_pos[1] < *(upper_y + i)) {
                bool good_solution = false;

                for (int f = 0; f < n_floor_ranges; f++) {
                    if (mario_pos[1] >= lower_floor[f] && mario_pos[1] <= upper_floor[f]) {
                        good_solution = true;
                        break;
                    }
                }

                if (!good_solution) {
                    float floor_dist = 65536.0;
                    float speed = 65536.0 * sqrtf(x * x + z * z);

                    for (int f = 0; f < n_floor_ranges; f++) {
                        float f_dist = mario_pos[1] - lower_floor[f];

                        if (f_dist > 0) {
                            floor_dist = f_dist;
                        }
                        else {
                            break;
                        }
                    }

                    int falling_frames = (int)ceil((sqrt(2.0 * floor_dist + 1.0) + 1.0) / 2.0);

                    int closest_pu_dist = fmin(fmin(mario_pos[0] + pow(2, 31), pow(2, 31) - 1.0 - mario_pos[0]), fmin(mario_pos[2] + pow(2, 31), pow(2, 31) - 1.0 - mario_pos[2]));

                    if (closest_pu_dist >= speed / 4.0) {
                        int total_falling_frames = (int)floor((pow(2, 32) - closest_pu_dist - 3.0 * speed / 2.0) / speed);

                        if (falling_frames <= total_falling_frames) {
                            good_solution = true;
                        }
                    }
                }


                if (good_solution) {
                    int solIdx = atomicAdd(&nUpwarpSolutions, 1);

                    if (solIdx < MAX_UPWARP_SOLUTIONS) {
                        UpwarpSolution solution;
                        solution.platformSolutionIdx = platSolIdx;
                        solution.upwarpPosition[0] = mario_pos[0];
                        solution.upwarpPosition[1] = mario_pos[1];
                        solution.upwarpPosition[2] = mario_pos[2];
                        solution.pux = (int)roundf(x / 65536.0f);
                        solution.puz = (int)roundf(z / 65536.0f);
                        upwarpSolutions[solIdx] = solution;
                    }

                    break;
                }
            }
        }
    }

    return true;
}

__device__ bool try_pu_x(float* normal, float* position, short(&current_triangles)[2][3][3], float(&triangle_normals)[2][3], float(&T_start)[4][4], float(&T_tilt)[4][4], double x, double x1_min, double x1_max, double x2_min, double x2_max, double platform_min_x, double platform_max_x, double platform_min_z, double platform_max_z, double m, double c_min, double c_max, int q_steps, float max_speed, int platSolIdx) {
    double pu_platform_min_x = x + platform_min_x;
    double pu_platform_max_x = x + platform_max_x;

    double pu_gap = 65536.0 * q_steps;

    // Find maximal range of PUs along z axis from current x PU position
    double min_z_pu_idx = (m * pu_platform_min_x + c_min) / pu_gap;
    double max_z_pu_idx = (m * pu_platform_max_x + c_max) / pu_gap;

    if (min_z_pu_idx > max_z_pu_idx) {
        double temp = min_z_pu_idx;
        min_z_pu_idx = max_z_pu_idx;
        max_z_pu_idx = temp;
    }

    // Check max_x_pu_idx and min_x_pu_idx are in range for valid platform tilt.
    // Correct them if they're not.
    //
    // Possible for only part of the platform to be in range.
    // In this case just skip whole PU to avoid headaches later on.

    if (pu_platform_max_x > fmin(x1_min, x1_max) && pu_platform_min_x < fmax(x1_min, x1_max)) {
        double z1_min = m * x1_min + c_min;
        double z1_max = m * x1_max + c_max;
        double tilt_cutoff_z = (z1_max - z1_min) * (x - x1_min) / (x1_max - x1_min) + z1_min;

        if (x1_min > 0) {
            // Find new lower bound for z_pu
            double tilt_cutoff_pu_idx = (tilt_cutoff_z - platform_max_z) / pu_gap;
            min_z_pu_idx = fmax(min_z_pu_idx, tilt_cutoff_pu_idx);
        }
        else {
            // Find new upper bound for z_pu
            double tilt_cutoff_pu_idx = (tilt_cutoff_z - platform_min_z) / pu_gap;
            max_z_pu_idx = fmin(max_z_pu_idx, tilt_cutoff_pu_idx);
        }
    }

    if (pu_platform_max_x > fmin(x2_min, x2_max) && pu_platform_min_x < fmax(x2_min, x2_max)) {
        double z2_min = m * x2_min + c_min;
        double z2_max = m * x2_max + c_max;
        double tilt_cutoff_z = (z2_max - z2_min) * (x - x2_min) / (x2_max - x2_min) + z2_min;

        if (x2_min > 0) {
            // Find new upper bound for z_pu
            double tilt_cutoff_pu_idx = (tilt_cutoff_z - platform_min_z) / pu_gap;
            max_z_pu_idx = fmin(max_z_pu_idx, tilt_cutoff_pu_idx);
        }
        else {
            // Find new lower bound for z_pu
            double tilt_cutoff_pu_idx = (tilt_cutoff_z - platform_max_z) / pu_gap;
            min_z_pu_idx = fmax(min_z_pu_idx, tilt_cutoff_pu_idx);
        }
    }

    min_z_pu_idx = q_steps * ceil(min_z_pu_idx);
    max_z_pu_idx = q_steps * floor(max_z_pu_idx);

    double min_z_pu = 65536.0 * min_z_pu_idx;
    double max_z_pu = 65536.0 * max_z_pu_idx;

    double closest_z_pu_platform;

    if (min_z_pu < 0) {
        if (max_z_pu < 0) {
            closest_z_pu_platform = max_z_pu + platform_max_z - platform_min_z;
        }
        else {
            if (abs(min_z_pu) < abs(max_z_pu)) {
                closest_z_pu_platform = min_z_pu + platform_max_z - platform_min_z;
            }
            else {
                closest_z_pu_platform = max_z_pu + platform_min_z - platform_max_z;
            }
        }
    }
    else {
        closest_z_pu_platform = min_z_pu + platform_min_z - platform_max_z;
    }

    // Find the minimum speed to reach a valid PU from current x position.
    // If this exceeds our maximum allowed speed, then we can stop searching polygon
    // in this direction.
    double min_needed_speed = (4.0 / (double)q_steps) * sqrt((x + platform_max_z - platform_min_z) * (x + platform_max_z - platform_min_z) + (closest_z_pu_platform * closest_z_pu_platform)) / fmax(triangle_normals[0][1], triangle_normals[1][1]);

    if (min_needed_speed > max_speed) {
        return false;
    }
    else {
        double min_pu_oob_z;

        if (q_steps < 4) {
            // If we're terminating Mario's movement early, then we need to be sure that 
            // there's enough of a difference between the y normals of the platform's two 
            // triangles to force Mario into out of bounds

            double closest_oob = 9743.23; // An estimate, based on the platforms pivot

            double min_dist_oob = closest_oob / (fmax(triangle_normals[0][1], triangle_normals[1][1]) / fmin(triangle_normals[0][1], triangle_normals[1][1]) - 1.0);
            double min_dist_oob_z = sqrt(min_dist_oob * min_dist_oob - x * x);

            min_pu_oob_z = ceil(min_dist_oob_z / 262144.0) * pu_gap;
        }
        else {
            min_pu_oob_z = 0.0;
        }

        double T_diff00 = T_tilt[0][0] - T_start[0][0];
        double T_diff20 = T_tilt[2][0] - T_start[2][0];
        double T_diff02 = T_tilt[0][2] - T_start[0][2];
        double T_diff22 = T_tilt[2][2] - T_start[2][2];

        // Tolerance for picking PUs that may result 
        // in out of bounds displacements.
        //
        // If we're more than the dimensions of the platform 
        // away from being in-bounds then we probably can't
        // get an in-bounds displacement anyway.
        double disp_leeway = abs(platform_min_x - platform_max_x) + abs(platform_min_z - platform_max_z);

        // Search backwards from z=0
        for (double z = fmin(fmin(0.0, max_z_pu), -min_pu_oob_z); z + 8192 > min_z_pu; z -= pu_gap) {
            double base_platform_displacement_x = x * T_diff00 + z * T_diff20;
            double base_platform_displacement_z = x * T_diff02 + z * T_diff22;

            double bpd_x_mod = static_cast<int16_t>(static_cast<int>(x + platform_pos[0] + base_platform_displacement_x));
            double bpd_z_mod = static_cast<int16_t>(static_cast<int>(z + platform_pos[2] + base_platform_displacement_z));

            // Check if our likely horizontal platform displacement puts us out of bounds.
            // If so, skip checking this PU.
            if (abs(bpd_x_mod) < 8192 + disp_leeway && abs(bpd_z_mod) < 8192 + disp_leeway) {
                if (!try_pu_xz(normal, position, current_triangles, triangle_normals, x, z, platSolIdx)) {
                    break;
                }
            }
        }

        // Search forwards from z>0
        for (double z = fmax(fmax(q_steps * pu_gap, min_z_pu), min_pu_oob_z); z - 8192 < max_z_pu; z += pu_gap) {
            double base_platform_displacement_x = x * T_diff00 + z * T_diff20;
            double base_platform_displacement_z = x * T_diff02 + z * T_diff22;

            double bpd_x_mod = static_cast<int16_t>(static_cast<int>(x + platform_pos[0] + base_platform_displacement_x));
            double bpd_z_mod = static_cast<int16_t>(static_cast<int>(z + platform_pos[2] + base_platform_displacement_z));

            // Check if our likely horizontal platform displacement puts us out of bounds.
            // If so, skip checking this PU.
            if (abs(bpd_x_mod) < 8192 + disp_leeway && abs(bpd_z_mod) < 8192 + disp_leeway) {
                if (!try_pu_xz(normal, position, current_triangles, triangle_normals, x, z, platSolIdx)) {
                    break;
                }
            }
        }

        return true;
    }
}

__device__ bool try_pu_z(float* normal, float* position, short(&current_triangles)[2][3][3], float(&triangle_normals)[2][3], float(&T_start)[4][4], float(&T_tilt)[4][4], double z, double z1_min, double z1_max, double z2_min, double z2_max, double platform_min_x, double platform_max_x, double platform_min_z, double platform_max_z, double m, double c_min, double c_max, int q_steps, float max_speed, int platSolIdx) {
    double pu_platform_min_z = z + platform_min_z;
    double pu_platform_max_z = z + platform_max_z;

    double pu_gap = 65535.0 * q_steps;

    // Find maximal range of PUs along x axis from current z PU position
    double min_x_pu_idx = ((pu_platform_min_z - c_min) / m) / pu_gap;
    double max_x_pu_idx = ((pu_platform_max_z - c_max) / m) / pu_gap;

    if (min_x_pu_idx > max_x_pu_idx) {
        double temp = min_x_pu_idx;
        min_x_pu_idx = max_x_pu_idx;
        max_x_pu_idx = temp;
    }

    // Check max_x_pu and min_x_pu are in range for valid platform tilt.
    // Correct them if they're not.
    //
    // Possible for only part of the platform to be in range.
    // In this case just skip it to avoid headaches later on.

    if (pu_platform_max_z > fmin(z1_min, z1_max) && pu_platform_min_z < fmax(z1_min, z1_max)) {
        double x1_min = (z1_min - c_min) / m;
        double x1_max = (z1_max - c_max) / m;
        double tilt_cutoff_x = (x1_max - x1_min) * (z - z1_min) / (z1_max - z1_min) + x1_min;

        if (z1_min > 0) {
            // Find new upper bound for z_pu
            double tilt_cutoff_pu_idx = (tilt_cutoff_x - platform_min_x) / pu_gap;
            max_x_pu_idx = fmin(max_x_pu_idx, tilt_cutoff_pu_idx);
        }
        else {
            // Find new lower bound for z_pu
            double tilt_cutoff_pu_idx = (tilt_cutoff_x - platform_max_x) / pu_gap;
            min_x_pu_idx = fmax(min_x_pu_idx, tilt_cutoff_pu_idx);
        }
    }

    if (pu_platform_max_z > fmin(z2_min, z2_max) && pu_platform_min_z < fmax(z2_min, z2_max)) {
        double x2_min = (z2_min - c_min) / m;
        double x2_max = (z2_max - c_max) / m;
        double tilt_cutoff_x = (x2_max - x2_min) * (z - z2_min) / (z2_max - z2_min) + x2_min;

        if (z2_min > 0) {
            // Find new lower bound for z_pu
            double tilt_cutoff_pu_idx = (tilt_cutoff_x - platform_max_x) / pu_gap;
            min_x_pu_idx = fmax(min_x_pu_idx, tilt_cutoff_pu_idx);
        }
        else {
            // Find new upper bound for z_pu
            double tilt_cutoff_pu_idx = (tilt_cutoff_x - platform_min_x) / pu_gap;
            max_x_pu_idx = fmin(max_x_pu_idx, tilt_cutoff_pu_idx);
        }
    }

    min_x_pu_idx = q_steps * ceil(min_x_pu_idx);
    max_x_pu_idx = q_steps * floor(max_x_pu_idx);

    double min_x_pu = 65536.0 * min_x_pu_idx;
    double max_x_pu = 65536.0 * max_x_pu_idx;

    double closest_x_pu_platform;

    if (min_x_pu < 0) {
        if (max_x_pu < 0) {
            closest_x_pu_platform = max_x_pu + platform_max_x - platform_min_x;
        }
        else {
            if (abs(min_x_pu) < abs(max_x_pu)) {
                closest_x_pu_platform = min_x_pu + platform_max_x - platform_min_x;
            }
            else {
                closest_x_pu_platform = max_x_pu + platform_min_x - platform_max_x;
            }
        }
    }
    else {
        closest_x_pu_platform = min_x_pu + platform_min_x - platform_max_x;
    }

    // Find the minimum speed to reach a valid PU from current z position.
    // If this exceeds our maximum allowed speed, then we can stop searching
    // the polygon in this direction.
    double min_needed_speed = (4.0 / (double)q_steps) * sqrt((z + platform_max_x - platform_min_x) * (z + platform_max_x - platform_min_x) + (closest_x_pu_platform * closest_x_pu_platform)) / fmax(triangle_normals[0][1], triangle_normals[1][1]);

    if (min_needed_speed > max_speed) {
        return false;
    }
    else {
        double min_pu_oob_x;

        if (q_steps < 4) {
            // If we're terminating Mario's movement early, then we need to be sure that 
            // there's enough of a difference between the y normals of the platform's two 
            // triangles to force Mario into out of bounds

            double closest_oob = 9743.23; // An estimate, based on the platform's pivot

            double min_dist_oob = closest_oob / (fmax(triangle_normals[0][1], triangle_normals[1][1]) / fmin(triangle_normals[0][1], triangle_normals[1][1]) - 1.0);
            double min_dist_oob_x = sqrt(min_dist_oob * min_dist_oob - z * z);

            min_pu_oob_x = ceil(min_dist_oob_x / 262144.0) * pu_gap;
        }
        else {
            min_pu_oob_x = 0.0;
        }

        double T_diff00 = T_tilt[0][0] - T_start[0][0];
        double T_diff20 = T_tilt[2][0] - T_start[2][0];
        double T_diff02 = T_tilt[0][2] - T_start[0][2];
        double T_diff22 = T_tilt[2][2] - T_start[2][2];

        // Tolerance for picking PUs that may result 
        // in out of bounds displacements.
        //
        // If we're more than the dimensions of the platform 
        // away from being in-bounds then we probably can't
        // get an in-bounds displacement anyway.
        double disp_leeway = abs(platform_min_x - platform_max_x) + abs(platform_min_z - platform_max_z);

        // Search backwards from x=0
        for (double x = fmin(fmin(0.0, max_x_pu), -min_pu_oob_x); x + 8192 > min_x_pu; x -= pu_gap) {
            double base_platform_displacement_x = x * T_diff00 + z * T_diff20;
            double base_platform_displacement_z = x * T_diff02 + z * T_diff22;

            double bpd_x_mod = static_cast<int16_t>(static_cast<int>(x + platform_pos[0] + base_platform_displacement_x));
            double bpd_z_mod = static_cast<int16_t>(static_cast<int>(z + platform_pos[2] + base_platform_displacement_z));

            // Check if our likely horizontal platform displacement puts us out of bounds.
            // If so, skip checking this PU.
            if (abs(bpd_x_mod) < 8192 + disp_leeway && abs(bpd_z_mod) < 8192 + disp_leeway) {
                if (!try_pu_xz(normal, position, current_triangles, triangle_normals, x, z, platSolIdx)) {
                    break;
                }
            }
        }

        // Search forwards from x>0
        for (double x = fmax(fmax(pu_gap, min_x_pu), min_pu_oob_x); x - 8192 < max_x_pu; x += pu_gap) {
            double base_platform_displacement_x = x * T_diff00 + z * T_diff20;
            double base_platform_displacement_z = x * T_diff02 + z * T_diff22;

            double bpd_x_mod = static_cast<int16_t>(static_cast<int>(x + platform_pos[0] + base_platform_displacement_x));
            double bpd_z_mod = static_cast<int16_t>(static_cast<int>(z + platform_pos[2] + base_platform_displacement_z));

            // Check if our likely horizontal platform displacement puts us out of bounds.
            // If so, skip checking this PU.
            if (abs(bpd_x_mod) < 8192 + disp_leeway && abs(bpd_z_mod) < 8192 + disp_leeway) {
                if (!try_pu_xz(normal, position, current_triangles, triangle_normals, x, z, platSolIdx)) {
                    break;
                }
            }
        }

        return true;
    }
}

__device__ void try_normal(float* normal, float* position, int platSolIdx, float max_speed) {
    // Tilt angle cut-offs
    // These are the yaw boundaries where the platform tilt 
    // switches direction. Directions match normal_offsets:
    // Between a[0] and a[1]: +x +z
    // Between a[1] and a[2]: -x +z
    // Between a[2] and a[3]: -x -z
    // Between a[3] and a[0]: +x -z

    short current_triangles[2][3][3];
    float triangle_normals[2][3];

    float T_start[4][4];
    T_start[1][0] = normal[0];
    T_start[1][1] = normal[1];
    T_start[1][2] = normal[2];

    float invsqrt = 1.0f / sqrtf(T_start[1][0] * T_start[1][0] + T_start[1][1] * T_start[1][1] + T_start[1][2] * T_start[1][2]);

    T_start[1][0] *= invsqrt;
    T_start[1][1] *= invsqrt;
    T_start[1][2] *= invsqrt;

    T_start[0][0] = T_start[1][1] * 1.0f - 0.0f * T_start[1][2];
    T_start[0][1] = T_start[1][2] * 0.0f - 1.0f * T_start[1][0];
    T_start[0][2] = T_start[1][0] * 0.0f - 0.0f * T_start[1][1];

    invsqrt = 1.0f / sqrtf(T_start[0][0] * T_start[0][0] + T_start[0][1] * T_start[0][1] + T_start[0][2] * T_start[0][2]);

    T_start[0][0] *= invsqrt;
    T_start[0][1] *= invsqrt;
    T_start[0][2] *= invsqrt;

    T_start[2][0] = T_start[0][1] * T_start[1][2] - T_start[1][1] * T_start[0][2];
    T_start[2][1] = T_start[0][2] * T_start[1][0] - T_start[1][2] * T_start[0][0];
    T_start[2][2] = T_start[0][0] * T_start[1][1] - T_start[1][0] * T_start[0][1];

    invsqrt = 1.0f / sqrtf(T_start[2][0] * T_start[2][0] + T_start[2][1] * T_start[2][1] + T_start[2][2] * T_start[2][2]);

    T_start[2][0] *= invsqrt;
    T_start[2][1] *= invsqrt;
    T_start[2][2] *= invsqrt;

    T_start[3][0] = platform_pos[0];
    T_start[3][1] = platform_pos[1];
    T_start[3][2] = platform_pos[2];
    T_start[0][3] = 0.0f;
    T_start[1][3] = 0.0f;
    T_start[2][3] = 0.0f;
    T_start[3][3] = 1.0f;

    for (int h = 0; h < 2; h++) {
        for (int i = 0; i < 3; i++) {
            float vx = default_triangles[h][i][0];
            float vy = default_triangles[h][i][1];
            float vz = default_triangles[h][i][2];

            current_triangles[h][i][0] = (short)(int)(vx * T_start[0][0] + vy * T_start[1][0] + vz * T_start[2][0] + T_start[3][0]);
            current_triangles[h][i][1] = (short)(int)(vx * T_start[0][1] + vy * T_start[1][1] + vz * T_start[2][1] + T_start[3][1]);
            current_triangles[h][i][2] = (short)(int)(vx * T_start[0][2] + vy * T_start[1][2] + vz * T_start[2][2] + T_start[3][2]);
        }

        triangle_normals[h][0] = ((current_triangles[h][1][1] - current_triangles[h][0][1]) * (current_triangles[h][2][2] - current_triangles[h][1][2])) - ((current_triangles[h][1][2] - current_triangles[h][0][2]) * (current_triangles[h][2][1] - current_triangles[h][1][1]));
        triangle_normals[h][1] = ((current_triangles[h][1][2] - current_triangles[h][0][2]) * (current_triangles[h][2][0] - current_triangles[h][1][0])) - ((current_triangles[h][1][0] - current_triangles[h][0][0]) * (current_triangles[h][2][2] - current_triangles[h][1][2]));
        triangle_normals[h][2] = ((current_triangles[h][1][0] - current_triangles[h][0][0]) * (current_triangles[h][2][1] - current_triangles[h][1][1])) - ((current_triangles[h][1][1] - current_triangles[h][0][1]) * (current_triangles[h][2][0] - current_triangles[h][1][0]));

        invsqrt = 1.0f / sqrtf(triangle_normals[h][0] * triangle_normals[h][0] + triangle_normals[h][1] * triangle_normals[h][1] + triangle_normals[h][2] * triangle_normals[h][2]);

        triangle_normals[h][0] *= invsqrt;
        triangle_normals[h][1] *= invsqrt;
        triangle_normals[h][2] *= invsqrt;
    }

    float nx = normal[0];
    float ny = normal[1];
    float nz = normal[2];

    double a[4];
    a[0] = atan2(nz, sqrt(1 - nz * nz));
    a[1] = atan2(sqrt(1 - nx * nx), nx);
    a[2] = M_PI - a[0];
    a[3] = 2 * M_PI - a[1];

    double platform_min_x = fmin(fmin((double)current_triangles[0][0][0], (double)current_triangles[0][1][0]), fmin((double)current_triangles[0][2][0], (double)current_triangles[1][2][0]));
    double platform_max_x = fmax(fmax((double)current_triangles[0][0][0], (double)current_triangles[0][1][0]), fmax((double)current_triangles[0][2][0], (double)current_triangles[1][2][0]));
    double platform_min_z = fmin(fmin((double)current_triangles[0][0][2], (double)current_triangles[0][1][2]), fmin((double)current_triangles[0][2][2], (double)current_triangles[1][2][2]));
    double platform_max_z = fmax(fmax((double)current_triangles[0][0][2], (double)current_triangles[0][1][2]), fmax((double)current_triangles[0][2][2], (double)current_triangles[1][2][2]));

    double min_y = fmin(-3071.0, fmin(fmin((double)current_triangles[0][0][1], (double)current_triangles[0][1][1]), fmin((double)current_triangles[0][2][1], (double)current_triangles[1][2][1])));
    double max_y = fmax(fmax((double)current_triangles[0][0][1], (double)current_triangles[0][1][1]), fmax((double)current_triangles[0][2][1], (double)current_triangles[1][2][1]));

    // Try to find solutions for each possible platform tilt direction
    for (int i = 0; i < 4; i++) {
        float T_tilt[4][4];
        T_tilt[1][0] = normal[0] + normal_offsets[i][0];
        T_tilt[1][1] = normal[1] + normal_offsets[i][1];
        T_tilt[1][2] = normal[2] + normal_offsets[i][2];

        float invsqrt = 1.0f / sqrtf(T_tilt[1][0] * T_tilt[1][0] + T_tilt[1][1] * T_tilt[1][1] + T_tilt[1][2] * T_tilt[1][2]);

        T_tilt[1][0] *= invsqrt;
        T_tilt[1][1] *= invsqrt;
        T_tilt[1][2] *= invsqrt;

        T_tilt[0][0] = T_tilt[1][1] * 1.0f - 0.0f * T_tilt[1][2];
        T_tilt[0][1] = T_tilt[1][2] * 0.0f - 1.0f * T_tilt[1][0];
        T_tilt[0][2] = T_tilt[1][0] * 0.0f - 0.0f * T_tilt[1][1];

        invsqrt = 1.0f / sqrtf(T_tilt[0][0] * T_tilt[0][0] + T_tilt[0][1] * T_tilt[0][1] + T_tilt[0][2] * T_tilt[0][2]);

        T_tilt[0][0] *= invsqrt;
        T_tilt[0][1] *= invsqrt;
        T_tilt[0][2] *= invsqrt;

        T_tilt[2][0] = T_tilt[0][1] * T_tilt[1][2] - T_tilt[1][1] * T_tilt[0][2];
        T_tilt[2][1] = T_tilt[0][2] * T_tilt[1][0] - T_tilt[1][2] * T_tilt[0][0];
        T_tilt[2][2] = T_tilt[0][0] * T_tilt[1][1] - T_tilt[1][0] * T_tilt[0][1];

        invsqrt = 1.0f / sqrtf(T_tilt[2][0] * T_tilt[2][0] + T_tilt[2][1] * T_tilt[2][1] + T_tilt[2][2] * T_tilt[2][2]);

        T_tilt[2][0] *= invsqrt;
        T_tilt[2][1] *= invsqrt;
        T_tilt[2][2] *= invsqrt;

        T_tilt[3][0] = platform_pos[0];
        T_tilt[3][1] = platform_pos[1];
        T_tilt[3][2] = platform_pos[2];
        T_tilt[0][3] = 0.0f;
        T_tilt[1][3] = 0.0f;
        T_tilt[2][3] = 0.0f;
        T_tilt[3][3] = 1.0f;

        double T_diff01 = T_tilt[0][1] - T_start[0][1];
        double T_diff11 = T_tilt[1][1] - T_start[1][1];
        double T_diff21 = T_tilt[2][1] - T_start[2][1];

        for (int j = 0; j < n_y_ranges; j++) {
            double r_min = lower_y[j] - (1 + T_diff11) * max_y + T_diff01 * platform_pos[0] + T_diff11 * platform_pos[1] + T_diff21 * platform_pos[2];
            double r_max = upper_y[j] - (1 + T_diff11) * min_y + T_diff01 * platform_pos[0] + T_diff11 * platform_pos[1] + T_diff21 * platform_pos[2];

            // z = mx + c_min
            // z = mx + c_max
            //
            // PU platforms occurring between these lines will (usually) 
            // give a y displacement within our desired range.
            double m = -T_diff01 / T_diff21;
            double c_min; double c_max;

            if (T_diff21 < 0) {
                c_min = r_max / T_diff21;
                c_max = r_min / T_diff21;
            }
            else {
                c_min = r_min / T_diff21;
                c_max = r_max / T_diff21;
            }

            // Find intersection between y displacement lines and 
            // good platform tilt angle ranges.
            //
            // Intersection forms a polygon that may (or may not)
            // stretch to infinity in one direction.
            // 
            // Find the x coordinates where displacement lines and 
            // platform tilt lines intersect.
            //
            // Non-intersecting lines have x coordinate set to NaN. 
            double a1_cos = cos(a[i]);
            double a2_cos = cos(a[(i + 1) % 4]);

            double x1_min; double x1_max; double x2_min; double x2_max;

            if (nx == 0) {
                if (i % 2 == 0) {
                    x1_min = (c_min + tan(a[i]) * platform_pos[0] - platform_pos[2]) / (tan(a[i]) - m);
                    x1_max = (c_max + tan(a[i]) * platform_pos[0] - platform_pos[2]) / (tan(a[i]) - m);
                    x2_min = 0;
                    x2_max = 0;

                    if (a1_cos > 0 && x1_min < platform_pos[0] || a1_cos < 0 && x1_min > platform_pos[0]) {
                        x1_min = NAN;
                    }

                    if (a1_cos > 0 && x1_max < platform_pos[0] || a1_cos < 0 && x1_max > platform_pos[0]) {
                        x1_max = NAN;
                    }

                    if (nz > 0 && c_min < platform_pos[0] || nz < 0 && c_min > platform_pos[0]) {
                        x2_min = NAN;
                    }

                    if (nz > 0 && c_max < platform_pos[0] || nz < 0 && c_max > platform_pos[0]) {
                        x2_max = NAN;
                    }
                }
                else {
                    x1_min = 0;
                    x1_max = 0;
                    x2_min = (c_min + tan(a[(i + 1) % 4]) * platform_pos[0] - platform_pos[2]) / (tan(a[(i + 1) % 4]) - m);
                    x2_max = (c_max + tan(a[(i + 1) % 4]) * platform_pos[0] - platform_pos[2]) / (tan(a[(i + 1) % 4]) - m);

                    if (nz > 0 && c_min < platform_pos[0] || nz < 0 && c_min > platform_pos[0]) {
                        x1_min = NAN;
                    }

                    if (nz > 0 && c_max < platform_pos[0] || nz < 0 && c_max > platform_pos[0]) {
                        x1_max = NAN;
                    }

                    if (a2_cos > 0 && x2_min < platform_pos[0] || a2_cos < 0 && x2_min > platform_pos[0]) {
                        x2_min = NAN;
                    }

                    if (a2_cos > 0 && x2_max < platform_pos[0] || a2_cos < 0 && x2_max >platform_pos[0]) {
                        x2_max = NAN;
                    }
                }
            }
            else {
                x1_min = (c_min + tan(a[i]) * platform_pos[0] - platform_pos[2]) / (tan(a[i]) - m);
                x1_max = (c_max + tan(a[i]) * platform_pos[0] - platform_pos[2]) / (tan(a[i]) - m);
                x2_min = (c_min + tan(a[(i + 1) % 4]) * platform_pos[0] - platform_pos[2]) / (tan(a[(i + 1) % 4]) - m);
                x2_max = (c_max + tan(a[(i + 1) % 4]) * platform_pos[0] - platform_pos[2]) / (tan(a[(i + 1) % 4]) - m);

                if (a1_cos > 0 && x1_min < platform_pos[0] || a1_cos < 0 && x1_min > platform_pos[0]) {
                    x1_min = NAN;
                }

                if (a1_cos > 0 && x1_max < platform_pos[0] || a1_cos < 0 && x1_max > platform_pos[0]) {
                    x1_max = NAN;
                }

                if (a2_cos > 0 && x2_min < platform_pos[0] || a2_cos < 0 && x2_min > platform_pos[0]) {
                    x2_min = NAN;
                }

                if (a2_cos > 0 && x2_max < platform_pos[0] || a2_cos < 0 && x2_max > platform_pos[0]) {
                    x2_max = NAN;
                }
            }


            // Mario's movement can end on any of his quarter steps, as long as the next move puts him 
            // out of bounds (or is the last step). So we need to consider PU movement for each possible
            // final quarter step

            // If the normals match then you can't force Mario out of bounds after his final q step.
            // Therefore, only 4 q_steps are possible.
            int q = 4;
            double pu_gap = 65536.0 * q;

            // Start searching for PUs in the polygon.
            //
            // We want to minimise speed, so we search outwards
            // from the point closest to the real platform.
            //
            // This will be at the x = 0 (if abs(m) < 1)
            // or z = 0 (if abs(m) > 1)
            if (abs(m) < 1) {
                // Find x limits of polygon
                double poly_x_start; double poly_x_end;

                if (!isnan(x1_min) && !isnan(x1_max)) {
                    if (!isnan(x2_min) && !isnan(x2_max)) {
                        poly_x_start = fmin(fmin(x1_min, x1_max), fmin(x2_min, x2_max));
                        poly_x_end = fmax(fmax(x1_min, x1_max), fmax(x2_min, x2_max));
                    }
                    else {
                        if (c_min > 0) {
                            poly_x_start = -INFINITY;
                            poly_x_end = fmax(x1_min, x1_max);
                        }
                        else {
                            poly_x_start = fmin(x1_min, x1_max);
                            poly_x_end = INFINITY;
                        }
                    }
                }
                else if (!isnan(x2_min) && !isnan(x2_max)) {
                    if (c_min > 0) {
                        poly_x_start = fmin(x2_min, x2_max);
                        poly_x_end = INFINITY;
                    }
                    else {
                        poly_x_start = -INFINITY;
                        poly_x_end = fmax(x2_min, x2_max);
                    }
                }
                else {
                    continue;
                }

                double first_x_pu = ceil((poly_x_start - platform_max_x) / pu_gap) * pu_gap;
                double last_x_pu = floor((poly_x_end - platform_min_x) / pu_gap) * pu_gap;

                // Search backwards from x=0
                for (double x = fmin(0.0, last_x_pu); x + platform_min_x > poly_x_start; x -= pu_gap) {
                    if (!try_pu_x(normal, position, current_triangles, triangle_normals, T_start, T_tilt, x, x1_min, x1_max, x2_min, x2_max, platform_min_x, platform_max_x, platform_min_z, platform_max_z, m, c_min, c_max, q, max_speed, platSolIdx)) {
                        break;
                    }
                }

                // Search forwards from x>0
                for (double x = fmax(pu_gap, first_x_pu); x - platform_max_x < poly_x_end; x += pu_gap) {
                    if (!try_pu_x(normal, position, current_triangles, triangle_normals, T_start, T_tilt, x, x1_min, x1_max, x2_min, x2_max, platform_min_x, platform_max_x, platform_min_z, platform_max_z, m, c_min, c_max, q, max_speed, platSolIdx)) {
                        break;
                    }
                }
            }
            else {
                // Calculate z coordinates of intersection points
                double z1_min = tan(a[i]) * x1_min + platform_pos[2] - tan(a[i]) * platform_pos[0];
                double z1_max = tan(a[i]) * x1_max + platform_pos[2] - tan(a[i]) * platform_pos[0];
                double z2_min = tan(a[(i + 1) % 4]) * x2_min + platform_pos[2] - tan(a[(i + 1) % 4]) * platform_pos[0];
                double z2_max = tan(a[(i + 1) % 4]) * x2_max + platform_pos[2] - tan(a[(i + 1) % 4]) * platform_pos[0];

                // Find z limits of polygon
                double poly_z_start; double poly_z_end;

                if (!isnan(z1_min) && !isnan(z1_max)) {
                    if (!isnan(z2_min) && !isnan(z2_max)) {
                        poly_z_start = fmin(fmin(z1_min, z1_max), fmin(z2_min, z2_max));
                        poly_z_end = fmax(fmax(z1_min, z1_max), fmax(z2_min, z2_max));
                    }
                    else {
                        if (c_min / m > 0) {
                            poly_z_start = -INFINITY;
                            poly_z_end = fmax(z1_min, z1_max);
                        }
                        else {
                            poly_z_start = fmin(z1_min, z1_max);
                            poly_z_end = INFINITY;
                        }
                    }
                }
                else if (!isnan(z2_min) && !isnan(z2_max)) {
                    if (c_min / m > 0) {
                        poly_z_start = fmin(z2_min, z2_max);
                        poly_z_end = INFINITY;
                    }
                    else {
                        poly_z_start = -INFINITY;
                        poly_z_end = fmax(z2_min, z2_max);
                    }
                }
                else {
                    continue;
                }

                double first_z_pu = ceil((poly_z_start - platform_max_z) / pu_gap) * pu_gap;
                double last_z_pu = floor((poly_z_end - platform_min_z) / pu_gap) * pu_gap;

                // Search backwards from z=0
                for (double z = fmin(0.0, last_z_pu); z + platform_min_z > poly_z_start; z -= pu_gap) {
                    if (!try_pu_z(normal, position, current_triangles, triangle_normals, T_start, T_tilt, z, z1_min, z1_max, z2_min, z2_max, platform_min_x, platform_max_x, platform_min_z, platform_max_z, m, c_min, c_max, q, max_speed, platSolIdx)) {
                        break;
                    }
                }

                // Search forwards from z>0
                for (double z = fmax(pu_gap, first_z_pu); z - platform_max_z < poly_z_end; z += pu_gap) {
                    if (!try_pu_z(normal, position, current_triangles, triangle_normals, T_start, T_tilt, z, z1_min, z1_max, z2_min, z2_max, platform_min_x, platform_max_x, platform_min_z, platform_max_z, m, c_min, c_max, q, max_speed, platSolIdx)) {
                        break;
                    }
                }
            }
        }
    }
}

__global__ void find_upwarp_solutions(float maxSpeed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < min(nPlatSolutions, MAX_PLAT_SOLUTIONS)) {
        struct PlatformSolution* platSol = &(platSolutions[idx]);
        try_normal(platSol->endNormal, platSol->endPosition, idx, maxSpeed);
    }
}

// Checks to see if an upwarp solution with the correct upwarpPosition was found.
__global__ void check_upwarp_solutions_for_the_right_one()
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nUpwarpSolutions, MAX_UPWARP_SOLUTIONS)) {
        struct UpwarpSolution* upSol = &(upwarpSolutions[idx]);

        if (abs(upSol->upwarpPosition[0] - 48434904) < 0.1 && abs(upSol->upwarpPosition[1] - 357.7861328) < 0.1 && abs(upSol->upwarpPosition[2] + 216856480) < 0.1 && upSol->pux == 752 && upSol->puz == -3304)
        {
            printf("Upwarp Solution found!\n");
        }
    }
}