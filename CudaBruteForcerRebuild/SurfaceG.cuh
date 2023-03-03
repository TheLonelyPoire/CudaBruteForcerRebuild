#pragma once

#include <fstream>
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

    __device__ SurfaceG(short x0, short y0, short z0, short x1, short y1, short z1, short x2, short y2, short z2);

    __device__ SurfaceG(short verts[3][3]);

    __device__ SurfaceG();

    __device__ void set_vertices(short verts[3][3]);

    __device__ void calculate_normal();
};