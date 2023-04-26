# pragma once

#include "cuda_runtime.h"


class SurfaceG {
public:
    short vertices[3][3];
    float normal[3];
    float origin_offset;
    float lower_y;
    float upper_y;

    float min_x;
    float max_x;
    float min_z;
    float max_z;

    bool is_lava;

    __device__ SurfaceG(short x0, short y0, short z0, short x1, short y1, short z1, short x2, short y2, short z2, bool lava) {
        short verts[3][3] = { {x0, y0, z0}, {x1, y1, z1}, {x2, y2, z2} };
        set_vertices(verts);
        is_lava = lava;
    }

    __device__ SurfaceG(short x0, short y0, short z0, short x1, short y1, short z1, short x2, short y2, short z2) {
        short verts[3][3] = { {x0, y0, z0}, {x1, y1, z1}, {x2, y2, z2} };
        set_vertices(verts);
        is_lava = false;
    }

    __device__ SurfaceG(short verts[3][3]) {
        set_vertices(verts);
    }

    __device__ SurfaceG() {}

    __device__ void set_vertices(short verts[3][3]) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                vertices[i][j] = verts[i][j];
            }
        }

        lower_y = fminf(fminf(vertices[0][1], vertices[1][1]), vertices[2][1]) - 5;
        upper_y = fmaxf(fmaxf(vertices[0][1], vertices[1][1]), vertices[2][1]) + 5;

        min_x = fminf(fminf(vertices[0][0], vertices[1][0]), vertices[2][0]);
        max_x = fmaxf(fmaxf(vertices[0][0], vertices[1][0]), vertices[2][0]);
        min_z = fminf(fminf(vertices[0][2], vertices[1][2]), vertices[2][2]);
        max_z = fmaxf(fmaxf(vertices[0][2], vertices[1][2]), vertices[2][2]);

        calculate_normal();
    }

    __device__ void calculate_normal() {
        normal[0] = (vertices[1][1] - vertices[0][1]) * (vertices[2][2] - vertices[1][2]) - (vertices[1][2] - vertices[0][2]) * (vertices[2][1] - vertices[1][1]);
        normal[1] = (vertices[1][2] - vertices[0][2]) * (vertices[2][0] - vertices[1][0]) - (vertices[1][0] - vertices[0][0]) * (vertices[2][2] - vertices[1][2]);
        normal[2] = (vertices[1][0] - vertices[0][0]) * (vertices[2][1] - vertices[1][1]) - (vertices[1][1] - vertices[0][1]) * (vertices[2][0] - vertices[1][0]);

        float mag = sqrtf(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);

        mag = (float)(1.0 / mag);
        normal[0] *= mag;
        normal[1] *= mag;
        normal[2] *= mag;

        origin_offset = -(normal[0] * vertices[0][0] + normal[1] * vertices[0][1] + normal[2] * vertices[0][2]);
    }
};

__device__ const int total_floorsG = 350;

extern __device__ SurfaceG floorsG[total_floorsG];

__global__ void initialise_floors();

__device__ bool fast_floor_check(float* intendedPosition, float& floorHeight, float& currentNormalY);


/**
 * Looks for a triangle of the two triangle inputs that is beneath the specified position.
 *
 * If one is found, the floor height pointer input is modified to the height of the triangle at the
 * specified position's X and Z coordinates, and the triangle's index in the specified set of triangles
 * is returned. Otherwise, an index of -1 is returned.
 *
 * @param pos - The position beneath which to check for floor triangles.
 * @param triangles - The set of triangles to check for floors.
 * @param normals - The corresponding set of normals for the set of triangles.
 * @param pheight - The pointer in which a found floor's height will be stored.
 * @return The index (either 0 or 1) of the floor triangle in the specified set of triangles, or -1 if no floor is found.
 */
__device__ int find_floor_triangles(float* pos, short(&triangles)[2][3][3], float(&normals)[2][3], float* pheight);


/**
 * Looks for a floor in the specified set of floors that is beneath the specified position.
 *
 * If one is found, the floor pointer input is modified to the found floor, the floor y pointer input is modified to the y position
 * of the floor, and the floor's index in the specified set of floors is returned. Otherwise, an index of -1 is returned.
 *
 * @param position - The position beneath which to check for floors.
 * @param floor - The pointer in which a found floor will be stored.
 * @param floor_y - The pointer in which a found floor's height will be stored.
 * @param floor_set - The set of floors to search.
 * @param n_floor_set - The total number of floors in the set of floors.
 * @return The index of the found floor in the set of floors, or -1 if no floor is found.
 */
__device__ int find_floor(float* position, SurfaceG** floor, float& floor_y, SurfaceG floor_set[], int n_floor_set);