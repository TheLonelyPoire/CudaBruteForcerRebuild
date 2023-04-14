#include "PlatformSolutionFunctions.cuh"

#include "math.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#include "BruteforceVariables.cuh"
#include "Floors.cuh"
#include "RunParameters.hpp"
#include "SolutionStructs.cuh"



__device__ void try_position(float* marioPos, float* normal, int maxFrames) {

    float returnNormal[3];
    returnNormal[0] = normal[0];
    returnNormal[1] = normal[1];
    returnNormal[2] = normal[2];

    const float platformPos[3] = { platform_pos[0], platform_pos[1], platform_pos[2] };
    const short defaultTriangles[2][3][3] = { {{307, 307, -306}, {-306, 307, -306}, {-306, 307, 307}}, {{307, 307, -306}, {-306, 307, 307}, {307, 307, 307}} };

    float mat[4][4];
    mat[1][0] = (normal[0] <= 0.0f) ? ((0.0f - normal[0] < 0.01f) ? 0.0f : (normal[0] + 0.01f)) : ((0.0f - normal[0] > -0.01f) ? 0.0f : (normal[0] - 0.01f));
    mat[1][1] = (normal[1] <= 1.0f) ? ((1.0f - normal[1] < 0.01f) ? 1.0f : (normal[1] + 0.01f)) : ((1.0f - normal[1] > -0.01f) ? 1.0f : (normal[1] - 0.01f));
    mat[1][2] = (normal[2] <= 0.0f) ? ((0.0f - normal[2] < 0.01f) ? 0.0f : (normal[2] + 0.01f)) : ((0.0f - normal[2] > -0.01f) ? 0.0f : (normal[2] - 0.01f));

    float invsqrt = 1.0f / sqrtf(mat[1][0] * mat[1][0] + mat[1][1] * mat[1][1] + mat[1][2] * mat[1][2]);

    mat[1][0] *= invsqrt;
    mat[1][1] *= invsqrt;
    mat[1][2] *= invsqrt;

    mat[0][0] = mat[1][1] * 1.0f - 0.0f * mat[1][2];
    mat[0][1] = mat[1][2] * 0.0f - 1.0f * mat[1][0];
    mat[0][2] = mat[1][0] * 0.0f - 0.0f * mat[1][1];

    invsqrt = 1.0f / sqrtf(mat[0][0] * mat[0][0] + mat[0][1] * mat[0][1] + mat[0][2] * mat[0][2]);

    mat[0][0] *= invsqrt;
    mat[0][1] *= invsqrt;
    mat[0][2] *= invsqrt;

    mat[2][0] = mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2];
    mat[2][1] = mat[0][2] * mat[1][0] - mat[1][2] * mat[0][0];
    mat[2][2] = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];

    invsqrt = 1.0f / sqrtf(mat[2][0] * mat[2][0] + mat[2][1] * mat[2][1] + mat[2][2] * mat[2][2]);

    mat[2][0] *= invsqrt;
    mat[2][1] *= invsqrt;
    mat[2][2] *= invsqrt;

    mat[3][0] = platformPos[0];
    mat[3][1] = platformPos[1];
    mat[3][2] = platformPos[2];
    mat[0][3] = 0.0f;
    mat[1][3] = 0.0f;
    mat[2][3] = 0.0f;
    mat[3][3] = 1.0f;

    short currentTriangles[2][3][3];
    float triangleNormals[2][3];

    for (int h = 0; h < 2; h++) {
        for (int i = 0; i < 3; i++) {
            float vx = defaultTriangles[h][i][0];
            float vy = defaultTriangles[h][i][1];
            float vz = defaultTriangles[h][i][2];

            currentTriangles[h][i][0] = (short)(int)(vx * mat[0][0] + vy * mat[1][0] + vz * mat[2][0] + mat[3][0]);
            currentTriangles[h][i][1] = (short)(int)(vx * mat[0][1] + vy * mat[1][1] + vz * mat[2][1] + mat[3][1]);
            currentTriangles[h][i][2] = (short)(int)(vx * mat[0][2] + vy * mat[1][2] + vz * mat[2][2] + mat[3][2]);
        }

        triangleNormals[h][0] = ((currentTriangles[h][1][1] - currentTriangles[h][0][1]) * (currentTriangles[h][2][2] - currentTriangles[h][1][2])) - ((currentTriangles[h][1][2] - currentTriangles[h][0][2]) * (currentTriangles[h][2][1] - currentTriangles[h][1][1]));
        triangleNormals[h][1] = ((currentTriangles[h][1][2] - currentTriangles[h][0][2]) * (currentTriangles[h][2][0] - currentTriangles[h][1][0])) - ((currentTriangles[h][1][0] - currentTriangles[h][0][0]) * (currentTriangles[h][2][2] - currentTriangles[h][1][2]));
        triangleNormals[h][2] = ((currentTriangles[h][1][0] - currentTriangles[h][0][0]) * (currentTriangles[h][2][1] - currentTriangles[h][1][1])) - ((currentTriangles[h][1][1] - currentTriangles[h][0][1]) * (currentTriangles[h][2][0] - currentTriangles[h][1][0]));

        invsqrt = 1.0f / sqrtf(triangleNormals[h][0] * triangleNormals[h][0] + triangleNormals[h][1] * triangleNormals[h][1] + triangleNormals[h][2] * triangleNormals[h][2]);

        triangleNormals[h][0] *= invsqrt;
        triangleNormals[h][1] *= invsqrt;
        triangleNormals[h][2] *= invsqrt;
    }

    float floor_height = 0.0;
    int floor_idx = find_floor_triangles(marioPos, currentTriangles, triangleNormals, &floor_height);

    if (floor_idx != -1 && floor_height - 100.0f > -3071.0f && floor_height >= -2967.168)
    {
        marioPos[1] = floor_height;

        mat[1][0] = normal[0];
        mat[1][1] = normal[1];
        mat[1][2] = normal[2];

        invsqrt = 1.0f / sqrtf(mat[1][0] * mat[1][0] + mat[1][1] * mat[1][1] + mat[1][2] * mat[1][2]);

        mat[1][0] *= invsqrt;
        mat[1][1] *= invsqrt;
        mat[1][2] *= invsqrt;

        mat[0][0] = mat[1][1] * 1.0f - 0.0f * mat[1][2];
        mat[0][1] = mat[1][2] * 0.0f - 1.0f * mat[1][0];
        mat[0][2] = mat[1][0] * 0.0f - 0.0f * mat[1][1];

        invsqrt = 1.0f / sqrtf(mat[0][0] * mat[0][0] + mat[0][1] * mat[0][1] + mat[0][2] * mat[0][2]);

        mat[0][0] *= invsqrt;
        mat[0][1] *= invsqrt;
        mat[0][2] *= invsqrt;

        mat[2][0] = mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2];
        mat[2][1] = mat[0][2] * mat[1][0] - mat[1][2] * mat[0][0];
        mat[2][2] = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];

        invsqrt = 1.0f / sqrtf(mat[2][0] * mat[2][0] + mat[2][1] * mat[2][1] + mat[2][2] * mat[2][2]);

        mat[2][0] *= invsqrt;
        mat[2][1] *= invsqrt;
        mat[2][2] *= invsqrt;

        mat[3][0] = platformPos[0];
        mat[3][1] = platformPos[1];
        mat[3][2] = platformPos[2];
        mat[0][3] = 0.0f;
        mat[1][3] = 0.0f;
        mat[2][3] = 0.0f;
        mat[3][3] = 1.0f;

        for (int h = 0; h < 2; h++) {
            for (int i = 0; i < 3; i++) {
                float vx = defaultTriangles[h][i][0];
                float vy = defaultTriangles[h][i][1];
                float vz = defaultTriangles[h][i][2];

                currentTriangles[h][i][0] = (short)(int)(vx * mat[0][0] + vy * mat[1][0] + vz * mat[2][0] + mat[3][0]);
                currentTriangles[h][i][1] = (short)(int)(vx * mat[0][1] + vy * mat[1][1] + vz * mat[2][1] + mat[3][1]);
                currentTriangles[h][i][2] = (short)(int)(vx * mat[0][2] + vy * mat[1][2] + vz * mat[2][2] + mat[3][2]);
            }

            triangleNormals[h][0] = ((currentTriangles[h][1][1] - currentTriangles[h][0][1]) * (currentTriangles[h][2][2] - currentTriangles[h][1][2])) - ((currentTriangles[h][1][2] - currentTriangles[h][0][2]) * (currentTriangles[h][2][1] - currentTriangles[h][1][1]));
            triangleNormals[h][1] = ((currentTriangles[h][1][2] - currentTriangles[h][0][2]) * (currentTriangles[h][2][0] - currentTriangles[h][1][0])) - ((currentTriangles[h][1][0] - currentTriangles[h][0][0]) * (currentTriangles[h][2][2] - currentTriangles[h][1][2]));
            triangleNormals[h][2] = ((currentTriangles[h][1][0] - currentTriangles[h][0][0]) * (currentTriangles[h][2][1] - currentTriangles[h][1][1])) - ((currentTriangles[h][1][1] - currentTriangles[h][0][1]) * (currentTriangles[h][2][0] - currentTriangles[h][1][0]));

            invsqrt = 1.0f / sqrtf(triangleNormals[h][0] * triangleNormals[h][0] + triangleNormals[h][1] * triangleNormals[h][1] + triangleNormals[h][2] * triangleNormals[h][2]);

            triangleNormals[h][0] *= invsqrt;
            triangleNormals[h][1] *= invsqrt;
            triangleNormals[h][2] *= invsqrt;
        }

        float returnPos[3] = { marioPos[0], marioPos[1], marioPos[2] };

        bool oTiltingPyramidMarioOnPlatform = false;
        bool onPlatform = false;

        float lastYNormal = triangleNormals[floor_idx][1];
        float lastPos[3] = { marioPos[0], marioPos[1], marioPos[2] };

        float landingPositions[3][3];
        float landingNormalsY[3];

        for (int f = 0; f < maxFrames; f++) {
            float dx;
            float dy;
            float dz;
            float d;

            float dist[3];
            float posBeforeRotation[3];
            float posAfterRotation[3];

            // Mario's position
            float mx;
            float my;
            float mz;

            int marioOnPlatform = 0;

            if (onPlatform)
            {
                mx = marioPos[0];
                my = marioPos[1];
                mz = marioPos[2];

                dist[0] = mx - (float)platformPos[0];
                dist[1] = my - (float)platformPos[1];
                dist[2] = mz - (float)platformPos[2];

                for (int i = 0; i < 3; i++) {
                    posBeforeRotation[i] = mat[0][i] * dist[0] + mat[1][i] * dist[1] + mat[2][i] * dist[2];
                }

                dx = mx - (float)platformPos[0];
                dy = 500.0f;
                dz = mz - (float)platformPos[2];
                d = sqrtf(dx * dx + dy * dy + dz * dz);

                //! Always true since dy = 500, making d >= 500.
                if (d != 0.0f) {
                    // Normalizing
                    d = 1.0 / d;
                    dx *= d;
                    dy *= d;
                    dz *= d;
                }
                else {
                    dx = 0.0f;
                    dy = 1.0f;
                    dz = 0.0f;
                }

                if (oTiltingPyramidMarioOnPlatform == true)
                    marioOnPlatform++;
                oTiltingPyramidMarioOnPlatform = true;
            }
            else
            {
                dx = 0.0f;
                dy = 1.0f;
                dz = 0.0f;
                oTiltingPyramidMarioOnPlatform = false;
            }

            // Approach the normals by 0.01f towards the new goal, then create a transform matrix and orient the object. 
            // Outside of the other conditionals since it needs to tilt regardless of whether Mario is on.

            normal[0] = (normal[0] <= dx) ? ((dx - normal[0] < 0.01f) ? dx : (normal[0] + 0.01f)) : ((dx - normal[0] > -0.01f) ? dx : (normal[0] - 0.01f));
            normal[1] = (normal[1] <= dy) ? ((dy - normal[1] < 0.01f) ? dy : (normal[1] + 0.01f)) : ((dy - normal[1] > -0.01f) ? dy : (normal[1] - 0.01f));
            normal[2] = (normal[2] <= dz) ? ((dz - normal[2] < 0.01f) ? dz : (normal[2] + 0.01f)) : ((dz - normal[2] > -0.01f) ? dz : (normal[2] - 0.01f));

            mat[1][0] = normal[0];
            mat[1][1] = normal[1];
            mat[1][2] = normal[2];

            invsqrt = 1.0f / sqrtf(mat[1][0] * mat[1][0] + mat[1][1] * mat[1][1] + mat[1][2] * mat[1][2]);

            mat[1][0] *= invsqrt;
            mat[1][1] *= invsqrt;
            mat[1][2] *= invsqrt;

            mat[0][0] = mat[1][1] * 1.0f - 0.0f * mat[1][2];
            mat[0][1] = mat[1][2] * 0.0f - 1.0f * mat[1][0];
            mat[0][2] = mat[1][0] * 0.0f - 0.0f * mat[1][1];

            invsqrt = 1.0f / sqrtf(mat[0][0] * mat[0][0] + mat[0][1] * mat[0][1] + mat[0][2] * mat[0][2]);

            mat[0][0] *= invsqrt;
            mat[0][1] *= invsqrt;
            mat[0][2] *= invsqrt;

            mat[2][0] = mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2];
            mat[2][1] = mat[0][2] * mat[1][0] - mat[1][2] * mat[0][0];
            mat[2][2] = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];

            invsqrt = 1.0f / sqrtf(mat[2][0] * mat[2][0] + mat[2][1] * mat[2][1] + mat[2][2] * mat[2][2]);

            mat[2][0] *= invsqrt;
            mat[2][1] *= invsqrt;
            mat[2][2] *= invsqrt;

            mat[3][0] = platformPos[0];
            mat[3][1] = platformPos[1];
            mat[3][2] = platformPos[2];
            mat[0][3] = 0.0f;
            mat[1][3] = 0.0f;
            mat[2][3] = 0.0f;
            mat[3][3] = 1.0f;

            for (int h = 0; h < 2; h++) {
                for (int i = 0; i < 3; i++) {
                    float vx = defaultTriangles[h][i][0];
                    float vy = defaultTriangles[h][i][1];
                    float vz = defaultTriangles[h][i][2];

                    currentTriangles[h][i][0] = (short)(int)(vx * mat[0][0] + vy * mat[1][0] + vz * mat[2][0] + mat[3][0]);
                    currentTriangles[h][i][1] = (short)(int)(vx * mat[0][1] + vy * mat[1][1] + vz * mat[2][1] + mat[3][1]);
                    currentTriangles[h][i][2] = (short)(int)(vx * mat[0][2] + vy * mat[1][2] + vz * mat[2][2] + mat[3][2]);
                }

                triangleNormals[h][0] = ((currentTriangles[h][1][1] - currentTriangles[h][0][1]) * (currentTriangles[h][2][2] - currentTriangles[h][1][2])) - ((currentTriangles[h][1][2] - currentTriangles[h][0][2]) * (currentTriangles[h][2][1] - currentTriangles[h][1][1]));
                triangleNormals[h][1] = ((currentTriangles[h][1][2] - currentTriangles[h][0][2]) * (currentTriangles[h][2][0] - currentTriangles[h][1][0])) - ((currentTriangles[h][1][0] - currentTriangles[h][0][0]) * (currentTriangles[h][2][2] - currentTriangles[h][1][2]));
                triangleNormals[h][2] = ((currentTriangles[h][1][0] - currentTriangles[h][0][0]) * (currentTriangles[h][2][1] - currentTriangles[h][1][1])) - ((currentTriangles[h][1][1] - currentTriangles[h][0][1]) * (currentTriangles[h][2][0] - currentTriangles[h][1][0]));

                invsqrt = 1.0f / sqrtf(triangleNormals[h][0] * triangleNormals[h][0] + triangleNormals[h][1] * triangleNormals[h][1] + triangleNormals[h][2] * triangleNormals[h][2]);

                triangleNormals[h][0] *= invsqrt;
                triangleNormals[h][1] *= invsqrt;
                triangleNormals[h][2] *= invsqrt;
            }

            // If Mario is on the platform, adjust his position for the platform tilt.
            if (marioOnPlatform) {
                for (int i = 0; i < 3; i++) {
                    posAfterRotation[i] = mat[0][i] * dist[0] + mat[1][i] * dist[1] + mat[2][i] * dist[2];
                }

                mx += posAfterRotation[0] - posBeforeRotation[0];
                my += posAfterRotation[1] - posBeforeRotation[1];
                mz += posAfterRotation[2] - posBeforeRotation[2];
                marioPos[0] = mx;
                marioPos[1] = my;
                marioPos[2] = mz;
            }

            floor_height = 0.0f;
            floor_idx = find_floor_triangles(marioPos, currentTriangles, triangleNormals, &floor_height);

            if (f < 3) {
                if (floor_idx == -1) {
                    landingNormalsY[f] = 1.0f;
                }
                else {
                    landingNormalsY[f] = triangleNormals[floor_idx][1];
                }

                landingPositions[f][0] = marioPos[0];
                landingPositions[f][1] = marioPos[1];
                landingPositions[f][2] = marioPos[2];
            }

            bool oldOnPlatform = onPlatform;
            onPlatform = floor_idx != -1 && fabsf(marioPos[1] - floor_height) <= 4.0;

            //Check if Mario is under the lava, or too far below the platform for it to conceivably be in reach later
            if ((floor_idx != -1 && floor_height <= -3071.0f) || (floor_idx != -1 && marioPos[1] - floor_height < -20.0f))
            {
                break;
            }

            if (onPlatform && oldOnPlatform) {
                float testNormal[3] = { fabs(normal[0]), fabs(normal[1]), fabs(normal[2]) };

                bool validSolution = false;

                if (testNormal[0] > testNormal[1] || testNormal[2] > testNormal[1]) {
                    validSolution = true;
                }
                else {
                    float offset = 0.01;

                    float a = testNormal[0] - offset;
                    float b = testNormal[2] - offset;
                    float c = testNormal[2];
                    float d = sqrtf(1 - testNormal[2] * testNormal[2]);
                    float sign = 1;

                    float v = testNormal[1] - offset;

                    float sqrt1 = sqrtf(a * a + v * v);
                    float sqrt2 = sqrtf(a * a + b * b + v * v);
                    float sqrt3 = sqrtf(testNormal[1] * testNormal[1] + testNormal[0] * testNormal[0]);
                    float sqrt4 = sqrtf(testNormal[1] * testNormal[1] + testNormal[0] * testNormal[0] + testNormal[2] * testNormal[2]);

                    float result = sign * d * sqrt1 * sqrt3 * (d * sqrt2 * (sqrt1 * testNormal[0] - a * sqrt3) * sqrt4 + c * (-sqrt1 * sqrt2 * testNormal[1] * testNormal[2] + b * v * sqrt3 * sqrt4));

                    if (result < 0) {
                        validSolution = true;
                    }
                    else {
                        c = sqrtf(1 - testNormal[0] * testNormal[0]);
                        d = testNormal[0];
                        sign = -1;
                        result = sign * d * sqrt1 * sqrt3 * (d * sqrt2 * (sqrt1 * testNormal[0] - a * sqrt3) * sqrt4 + c * (-sqrt1 * sqrt2 * testNormal[1] * testNormal[2] + b * v * sqrt3 * sqrt4));

                        if (result < 0) {
                            validSolution = true;
                        }
                    }
                }

                if (validSolution) {
                    int solIdx = atomicAdd(&nPlatSolutions, 1);

                    if (solIdx < MAX_PLAT_SOLUTIONS) {
                        struct PlatformSolution solution;
                        solution.endNormal[0] = normal[0];
                        solution.endNormal[1] = normal[1];
                        solution.endNormal[2] = normal[2];
                        solution.endPosition[0] = marioPos[0];
                        solution.endPosition[1] = marioPos[1];
                        solution.endPosition[2] = marioPos[2];
                        solution.returnPosition[0] = returnPos[0];
                        solution.returnPosition[1] = returnPos[1];
                        solution.returnPosition[2] = returnPos[2];
                        solution.returnNormal[0] = returnNormal[0];
                        solution.returnNormal[1] = returnNormal[1];
                        solution.returnNormal[2] = returnNormal[2];
                        solution.nFrames = f;
                        solution.penultimateFloorNormalY = lastYNormal;
                        solution.penultimatePosition[0] = lastPos[0];
                        solution.penultimatePosition[1] = lastPos[1];
                        solution.penultimatePosition[2] = lastPos[2];
                        for (int j = 0; j < 2; j++) {
                            solution.endTriangleNormals[j][0] = triangleNormals[j][0];
                            solution.endTriangleNormals[j][1] = triangleNormals[j][1];
                            solution.endTriangleNormals[j][2] = triangleNormals[j][2];

                            for (int k = 0; k < 3; k++) {
                                solution.endTriangles[j][k][0] = currentTriangles[j][k][0];
                                solution.endTriangles[j][k][1] = currentTriangles[j][k][1];
                                solution.endTriangles[j][k][2] = currentTriangles[j][k][2];
                            }
                        }
                        for (int f = 0; f < 3; f++) {
                            solution.landingPositions[f][0] = landingPositions[f][0];
                            solution.landingPositions[f][1] = landingPositions[f][1];
                            solution.landingPositions[f][2] = landingPositions[f][2];
                            solution.landingFloorNormalsY[f] = landingNormalsY[f];
                        }

                        platSolutions[solIdx] = solution;
                    }
                }
            }

            lastYNormal = triangleNormals[floor_idx][1];
            lastPos[0] = marioPos[0];
            lastPos[1] = marioPos[1];
            lastPos[2] = marioPos[2];
        }
    }
}

__global__ void cudaFunc(const float minX, const float deltaX, const float minZ, const float deltaZ, const int width, const int height, float normalX, float normalY, float normalZ, int maxFrames) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < width * height) {
        float marioPos[3] = { minX - fmodf(minX, deltaX) + deltaX * (idx % width), -2500.0f, minZ - fmodf(minZ, deltaZ) + deltaZ * (idx / width) };
        float normal[3] = { normalX, normalY, normalZ };

        try_position(marioPos, normal, maxFrames);
    }
}

// Checks to see if a platform solution with the correct endNormal was found.
__global__ void check_platform_solutions_for_the_right_one()
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nPlatSolutions, MAX_PLAT_SOLUTIONS)) {
        struct PlatformSolution* platSol = &(platSolutions[idx]);

        if (abs(platSol->endNormal[0] - 0.2320000678) < 0.0001 && abs(platSol->endNormal[1] - 0.8384857774) < 0.0001 && abs(platSol->endNormal[2] - 0.4581809938) < 0.0001)
        {
            printf("Platform Solution found!\n\n");
            printf("Index: %i \n\n", idx);
            printf("End Normal X: ");
            printf("%f", platSol->endNormal[0]);
            printf("\nEnd Normal Y: ");
            printf("%f", platSol->endNormal[1]);
            printf("\nEnd Normal Z: ");
            printf("%f", platSol->endNormal[2]);
            printf("\n");
            printf("\nEnd Position X: ");
            printf("%f", platSol->endPosition[0]);
            printf("\nEnd Position Y: ");
            printf("%f", platSol->endPosition[1]);
            printf("\nEnd Position Z: ");
            printf("%f", platSol->endPosition[2]);
            printf("\n");
            printf("\nNFrames: ");
            printf("%i", platSol->nFrames);
            printf("\n");
        }
    }
}