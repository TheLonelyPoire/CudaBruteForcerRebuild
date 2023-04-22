#include "HAUFunctions.cuh"

#include "math.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#include "Platform.cuh"
#include "vmath.hpp"

#include "BruteforceVariables.cuh"
#include "CommonFunctions.cuh"
#include "Floors.cuh"
#include "PlatformSolutionFunctions.cuh"
#include "RunParameters.hpp"
#include "UpwarpSolutionFunctions.cuh"


__global__ void copy_solution_pointers_hau(HAUSolStruct s) {
    platSolutions = s.platSolutions;
    upwarpSolutions = s.upwarpSolutions;
    stickSolutions = s.stickSolutions;
    oupSolutions = s.oupSolutions;
    speedSolutionsHAU = s.speedSolutions;
    tenKSolutionsHAU = s.tenKSolutions;
}

void init_solution_structs_hau(HAUSolStruct* s) {
    cudaMalloc((void**)&s->platSolutions, MAX_PLAT_SOLUTIONS * sizeof(PlatformSolution));
    cudaMalloc((void**)&s->upwarpSolutions, MAX_UPWARP_SOLUTIONS * sizeof(UpwarpSolution));
    cudaMalloc((void**)&s->stickSolutions, MAX_STICK_SOLUTIONS * sizeof(StickSolution));
    cudaMalloc((void**)&s->oupSolutions, MAX_OUP_SOLUTIONS * sizeof(OUPSolution));
    cudaMalloc((void**)&s->speedSolutions, MAX_SPEED_SOLUTIONS_HAU * sizeof(StickSolution));
    cudaMalloc((void**)&s->tenKSolutions, MAX_10K_SOLUTIONS_HAU * sizeof(TenKSolutionHAU));

    copy_solution_pointers_hau << <1, 1 >> > (*s);
}

void free_solution_pointers_hau(HAUSolStruct* s) {
    cudaFree(s->platSolutions);
    cudaFree(s->upwarpSolutions);
    cudaFree(s->stickSolutions);
    cudaFree(s->oupSolutions);
    cudaFree(s->speedSolutions);
    cudaFree(s->tenKSolutions);
}


__global__ void test_speed_solution() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nSpeedSolutionsHAU, MAX_SPEED_SOLUTIONS_HAU)) {
        struct SpeedSolutionHAU* speedSol = &(speedSolutionsHAU[idx]);
        struct OUPSolution* oupSol = &(oupSolutions[speedSol->oupSolutionIdx]);
        struct StickSolution* stickSol = &(stickSolutions[oupSol->stickSolutionIdx]);
        struct UpwarpSolution* uwSol = &(upwarpSolutions[stickSol->upwarpSolutionIdx]);
        struct PlatformSolution* platSol = &(platSolutions[uwSol->platformSolutionIdx]);

        float oneUpPlatformNormalX = stickSol->xDir == 0 ? oneUpPlatformNormalXRight : oneUpPlatformNormalXLeft;
        float oneUpPlatformNormalY = stickSol->xDir == 0 ? oneUpPlatformNormalYRight : oneUpPlatformNormalYLeft;
        float oneUpPlatformXMin = stickSol->xDir == 0 ? oneUpPlatformXMinRight : oneUpPlatformXMinLeft;
        float oneUpPlatformXMax = stickSol->xDir == 0 ? oneUpPlatformXMaxRight : oneUpPlatformXMaxLeft;
        float oneUpPlatformZMin = stickSol->xDir == 0 ? oneUpPlatformZMinRight : oneUpPlatformZMinLeft;
        float oneUpPlatformZMax = stickSol->xDir == 0 ? oneUpPlatformZMaxRight : oneUpPlatformZMaxLeft;
        float oneUpPlatformYMin = stickSol->xDir == 0 ? oneUpPlatformYMinRight : oneUpPlatformYMinLeft;
        float oneUpPlatformYMax = stickSol->xDir == 0 ? oneUpPlatformYMaxRight : oneUpPlatformYMaxLeft;

        float relY = stickSol->stickY + 6.0f;
        float intendedMag = (relY * relY / 128.0f);
        int intendedYaw = atan2sG(-relY, 0) + oupSol->cameraYaw;
        intendedYaw = (65536 + intendedYaw) % 65536;
        int intendedDYaw = (65536 + intendedYaw - oupSol->angle) % 65536;

        float lossFactor = gCosineTableG[intendedDYaw / 16];
        lossFactor *= 0.5f + 0.5f * speedSol->startSpeed / 100.0f;
        lossFactor = intendedMag / 32.0f * lossFactor * 0.02f + 0.92f;

        float startSpeedX = speedSol->startSpeed * gSineTableG[(oupSol->angle) / 16];
        float startSpeedZ = speedSol->startSpeed * gCosineTableG[(oupSol->angle) / 16];

        float returnSpeedX = startSpeedX;
        float returnSpeedZ = startSpeedZ;

        returnSpeedX += returnSpeedZ * (intendedMag / 32.0f) * gSineTableG[intendedDYaw / 16] * 0.05f;
        returnSpeedZ -= returnSpeedX * (intendedMag / 32.0f) * gSineTableG[intendedDYaw / 16] * 0.05f;

        float newSpeed = sqrtf(returnSpeedX * returnSpeedX + returnSpeedZ * returnSpeedZ);

        returnSpeedX = returnSpeedX * speedSol->startSpeed / newSpeed;
        returnSpeedZ = returnSpeedZ * speedSol->startSpeed / newSpeed;

        returnSpeedX += 7.0f * oneUpPlatformNormalX;

        returnSpeedX *= lossFactor;
        returnSpeedZ *= lossFactor;

        float returnSpeed = -sqrtf(returnSpeedX * returnSpeedX + returnSpeedZ * returnSpeedZ);

        float oneUpPlatformPosition[3];
        oneUpPlatformPosition[0] = platSol->returnPosition[0] - (oneUpPlatformNormalY + stickSol->q3 - 1.0) * returnSpeedX / 4.0f;
        oneUpPlatformPosition[2] = platSol->returnPosition[2] - (oneUpPlatformNormalY + stickSol->q3 - 1.0) * returnSpeedZ / 4.0f;

        float intendedPosition[3];
        intendedPosition[0] = oneUpPlatformPosition[0];
        intendedPosition[2] = oneUpPlatformPosition[2];

        float currentNormalY = oneUpPlatformNormalY;

        for (int j = 1; j <= stickSol->q3; j++) {
            intendedPosition[0] = intendedPosition[0] + currentNormalY * returnSpeedX / 4.0f;
            intendedPosition[2] = intendedPosition[2] + currentNormalY * returnSpeedZ / 4.0f;
            currentNormalY = 1.0f;
        }

        oneUpPlatformPosition[0] = oneUpPlatformPosition[0] - (intendedPosition[0] - platSol->returnPosition[0]);
        oneUpPlatformPosition[2] = oneUpPlatformPosition[2] - (intendedPosition[2] - platSol->returnPosition[2]);

        intendedPosition[0] = oneUpPlatformPosition[0];
        intendedPosition[2] = oneUpPlatformPosition[2];

        int returnSlideYaw = atan2sG(returnSpeedZ, returnSpeedX);
        int newFacingDYaw = (short)(oupSol->angle - returnSlideYaw);

        if (newFacingDYaw > 0 && newFacingDYaw <= 0x4000) {
            if ((newFacingDYaw -= 0x200) < 0) {
                newFacingDYaw = 0;
            }
        }
        else if (newFacingDYaw > -0x4000 && newFacingDYaw < 0) {
            if ((newFacingDYaw += 0x200) > 0) {
                newFacingDYaw = 0;
            }
        }
        else if (newFacingDYaw > 0x4000 && newFacingDYaw < 0x8000) {
            if ((newFacingDYaw += 0x200) > 0x8000) {
                newFacingDYaw = 0x8000;
            }
        }
        else if (newFacingDYaw > -0x8000 && newFacingDYaw < -0x4000) {
            if ((newFacingDYaw -= 0x200) < -0x8000) {
                newFacingDYaw = -0x8000;
            }
        }

        int returnFaceAngle = returnSlideYaw + newFacingDYaw;
        returnFaceAngle = (65536 + returnFaceAngle) % 65536;

        float postReturnVelX = returnSpeed * gSineTableG[returnFaceAngle >> 4];
        float postReturnVelZ = returnSpeed * gCosineTableG[returnFaceAngle >> 4];

        intendedPosition[0] = platSol->returnPosition[0] + postReturnVelX / 4.0;
        intendedPosition[1] = platSol->returnPosition[1];
        intendedPosition[2] = platSol->returnPosition[2] + postReturnVelZ / 4.0;

        bool outOfBoundsTest = !check_inbounds(intendedPosition);

        for (int f = 0; outOfBoundsTest && f < 3; f++) {
            intendedPosition[0] = platSol->landingPositions[f][0] + platSol->landingFloorNormalsY[f] * (postReturnVelX / 4.0);
            intendedPosition[1] = platSol->landingPositions[f][1];
            intendedPosition[2] = platSol->landingPositions[f][2] + platSol->landingFloorNormalsY[f] * (postReturnVelZ / 4.0);

            outOfBoundsTest = !check_inbounds(intendedPosition);
        }

        if (outOfBoundsTest && (short)(int)oneUpPlatformPosition[0] >= oneUpPlatformXMin && (short)(int)oneUpPlatformPosition[0] <= oneUpPlatformXMax && (short)(int)oneUpPlatformPosition[2] >= oneUpPlatformZMin && (short)(int)oneUpPlatformPosition[2] <= oneUpPlatformZMax) {
            oneUpPlatformPosition[1] = oneUpPlatformYMin + (oneUpPlatformYMax - oneUpPlatformYMin) * (((float)(short)(int)oneUpPlatformPosition[0] - oneUpPlatformXMin) / (oneUpPlatformXMax - oneUpPlatformXMin));

            bool fallTest = false;

            intendedPosition[0] = oneUpPlatformPosition[0];
            intendedPosition[1] = oneUpPlatformPosition[1];
            intendedPosition[2] = oneUpPlatformPosition[2];
            currentNormalY = oneUpPlatformNormalY;

            int i;

            for (i = 1; i <= 4; i++) {
                intendedPosition[0] = intendedPosition[0] + currentNormalY * returnSpeedX / 4.0f;
                intendedPosition[2] = intendedPosition[2] + currentNormalY * returnSpeedZ / 4.0f;

                SurfaceG* floor;
                float floorHeight;
                int floorIdx = find_floor(intendedPosition, &floor, floorHeight, floorsG, total_floorsG);

                if (floorIdx == -1) {
                    break;
                }
                else {
                    if (intendedPosition[1] <= floorHeight + 100.0f) {
                        currentNormalY = floor->normal[1];
                        intendedPosition[1] = floorHeight;
                    }
                    else {
                        if (i == stickSol->q3) {
                            fallTest = true;
                            // TODO - REMOVE
                            atomicAdd(&nPass1Sols, 1);
                        }

                        break;
                    }
                }
            }

            if (fallTest && intendedPosition[1] <= platSol->returnPosition[1] && intendedPosition[0] == platSol->returnPosition[0] && intendedPosition[0] == platSol->returnPosition[0] && intendedPosition[2] == platSol->returnPosition[2]) {
                // TODO - REMOVE
                atomicAdd(&nPass2Sols, 1);
                for (int q1 = max(1, stickSol->q1q2 - 4); q1 <= min(4, stickSol->q1q2 - 1); q1++) {
                    int q2 = stickSol->q1q2 - q1;

                    float frame1Position[3];
                    frame1Position[0] = oneUpPlatformPosition[0] - q2 * startSpeedX / 4.0f;
                    frame1Position[2] = oneUpPlatformPosition[2] - q2 * startSpeedZ / 4.0f;

                    float intendedPosition[3];
                    intendedPosition[0] = frame1Position[0];
                    intendedPosition[2] = frame1Position[2];

                    for (int j = 1; j <= q2; j++) {
                        intendedPosition[0] = intendedPosition[0] + startSpeedX / 4.0f;
                        intendedPosition[2] = intendedPosition[2] + startSpeedZ / 4.0f;
                    }

                    frame1Position[0] = frame1Position[0] - (intendedPosition[0] - oneUpPlatformPosition[0]);
                    frame1Position[2] = frame1Position[2] - (intendedPosition[2] - oneUpPlatformPosition[2]);

                    if ((short)(int)frame1Position[0] >= -8191 && (short)(int)frame1Position[0] <= 8192 && (short)(int)frame1Position[2] >= -8191 && (short)(int)frame1Position[2] <= 8192) {
                        float startPosition[3];
                        startPosition[0] = frame1Position[0] - (startNormals[stickSol->floorIdx][1] + q1 - 1.0) * startSpeedX / 4.0f;
                        startPosition[2] = frame1Position[2] - (startNormals[stickSol->floorIdx][1] + q1 - 1.0) * startSpeedZ / 4.0f;

                        float intendedPosition[3];
                        intendedPosition[0] = startPosition[0];
                        intendedPosition[2] = startPosition[2];

                        float currentNormalY = startNormals[stickSol->floorIdx][1];

                        for (int j = 1; j <= q1; j++) {
                            intendedPosition[0] = intendedPosition[0] + currentNormalY * startSpeedX / 4.0f;
                            intendedPosition[2] = intendedPosition[2] + currentNormalY * startSpeedZ / 4.0f;
                            currentNormalY = 1.0f;
                        }

                        startPosition[0] = startPosition[0] - (intendedPosition[0] - frame1Position[0]);
                        startPosition[2] = startPosition[2] - (intendedPosition[2] - frame1Position[2]);

                        float oo = -(startNormals[stickSol->floorIdx][0] * startTriangles[stickSol->floorIdx][0][0] + startNormals[stickSol->floorIdx][1] * startTriangles[stickSol->floorIdx][0][1] + startNormals[stickSol->floorIdx][2] * startTriangles[stickSol->floorIdx][0][2]);

                        startPosition[1] = -(startPosition[0] * startNormals[stickSol->floorIdx][0] + startNormals[stickSol->floorIdx][2] * startPosition[2] + oo) / startNormals[stickSol->floorIdx][1];

                        fallTest = false;

                        intendedPosition[0] = startPosition[0];
                        intendedPosition[1] = startPosition[1];
                        intendedPosition[2] = startPosition[2];
                        currentNormalY = startNormals[stickSol->floorIdx][1];

                        for (i = 1; i <= 4; i++) {
                            intendedPosition[0] = intendedPosition[0] + currentNormalY * startSpeedX / 4.0f;
                            intendedPosition[2] = intendedPosition[2] + currentNormalY * startSpeedZ / 4.0f;

                            SurfaceG* floor;
                            float floorHeight;
                            int floorIdx = find_floor(intendedPosition, &floor, floorHeight, floorsG, total_floorsG);

                            if (floorIdx == -1) {
                                break;
                            }
                            else {
                                if (intendedPosition[1] <= floorHeight + 100.0f) {
                                    currentNormalY = floor->normal[1];
                                    intendedPosition[1] = floorHeight;
                                }
                                else {
                                    if (i == q1) {
                                        fallTest = true;
                                        // TODO - REMOVE
                                        atomicAdd(&nPass3Sols, 1);
                                        break;
                                    }
                                    else {
                                        short relX = (short)(int)intendedPosition[0];
                                        short relZ = (short)(int)intendedPosition[2];

                                        if (relZ >= -306.0f && relZ <= 307.0f && relX >= -6041.0f && relX <= -306.0f) {
                                            currentNormalY = 1.0f;
                                            intendedPosition[1] = oneUpPlatformPosition[1] - 1.0f;
                                        }
                                        else {
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        atomicMinFloat(&currentLowestHeightDiff, intendedPosition[1] - oneUpPlatformPosition[1]);

                        if (fallTest && intendedPosition[1] <= oneUpPlatformPosition[1]) {
                            frame1Position[1] = intendedPosition[1];

                            printf("---------------------------------------\nFound Solution:\n---------------------------------------\n    Start Position: %.10g, %.10g, %.10g\n    Frame 1 Position: %.10g, %.10g, %.10g\n    Frame 2 Position: %.10g, %.10g, %.10g\n    Return Position: %.10g, %.10g, %.10g\n    PU Route Speed: %.10g (x=%.10g, z=%.10g)\n    PU Return Speed: %.10g (x=%.10g, z=%.10g)\n    Frame 1 Q-steps: %d\n    Frame 2 Q-steps: %d\n    Frame 3 Q-steps: %d\n", startPosition[0], startPosition[1], startPosition[2], frame1Position[0], frame1Position[1], frame1Position[2], oneUpPlatformPosition[0], oneUpPlatformPosition[1], oneUpPlatformPosition[2], platSol->returnPosition[0], platSol->returnPosition[1], platSol->returnPosition[2], speedSol->startSpeed, startSpeedX, startSpeedZ, returnSpeed, returnSpeedX, returnSpeedZ, q1, q2, stickSol->q3);
                            printf("    Frame 1 Angle: %d\n    10k Stick X: %d\n    10k Stick Y: %d\n    10k Camera Yaw: %d\n    Start Floor Normal: %.10g, %.10g, %.10g\n", oupSol->angle, 0, stickSol->stickY, oupSol->cameraYaw, startNormals[stickSol->floorIdx][0], startNormals[stickSol->floorIdx][1], startNormals[stickSol->floorIdx][2]);
                            printf("---------------------------------------\n    Tilt Frames: %d\n    Post-Tilt Platform Normal: %.10g, %.10g, %.10g\n    Post-Tilt Position: %.10g, %.10g, %.10g\n    Post-Upwarp Position: %.10g, %.10g, %.10g\n    Upwarp PU X: %d\n    Upwarp PU Z: %d\n---------------------------------------\n\n\n", platSol->nFrames, platSol->endNormal[0], platSol->endNormal[1], platSol->endNormal[2], platSol->endPosition[0], platSol->endPosition[1], platSol->endPosition[2], uwSol->upwarpPosition[0], uwSol->upwarpPosition[1], uwSol->upwarpPosition[2], uwSol->pux, uwSol->puz);

                            int tenKSolIdx = atomicAdd(&n10KSolutionsHAU, 1);

                            if (tenKSolIdx < MAX_10K_SOLUTIONS_HAU) {
                                struct TenKSolutionHAU solution;
                                solution.speedSolutionIdx = idx;
                                solution.startPosition[0] = startPosition[0];
                                solution.startPosition[1] = startPosition[1];
                                solution.startPosition[2] = startPosition[2];
                                solution.frame1Position[0] = frame1Position[0];
                                solution.frame1Position[1] = frame1Position[1];
                                solution.frame1Position[2] = frame1Position[2];
                                solution.frame2Position[0] = oneUpPlatformPosition[0];
                                solution.frame2Position[1] = oneUpPlatformPosition[1];
                                solution.frame2Position[2] = oneUpPlatformPosition[2];
                                solution.frame1QSteps = q1;
                                solution.frame2QSteps = q2;
                                solution.startSpeedX = startSpeedX;
                                solution.startSpeedZ = startSpeedZ;
                                solution.returnSpeed = returnSpeed;
                                solution.returnSpeedX = returnSpeedX;
                                solution.returnSpeedZ = returnSpeedZ;
                                tenKSolutionsHAU[tenKSolIdx] = solution;
                            }
                        }
                    }
                }
            }
        }
    }
}

__global__ void test_oup_solution() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nOUPSolutions, MAX_OUP_SOLUTIONS)) {
        struct OUPSolution* oupSol = &(oupSolutions[idx]);
        struct StickSolution* stickSol = &(stickSolutions[oupSol->stickSolutionIdx]);
        struct UpwarpSolution* uwSol = &(upwarpSolutions[stickSol->upwarpSolutionIdx]);
        struct PlatformSolution* platSol = &(platSolutions[uwSol->platformSolutionIdx]);

        float oneUpPlatformNormalX = stickSol->xDir == 0 ? oneUpPlatformNormalXRight : oneUpPlatformNormalXLeft;
        float oneUpPlatformNormalY = stickSol->xDir == 0 ? oneUpPlatformNormalYRight : oneUpPlatformNormalYLeft;
        float oneUpPlatformXMin = stickSol->xDir == 0 ? oneUpPlatformXMinRight : oneUpPlatformXMinLeft;
        float oneUpPlatformXMax = stickSol->xDir == 0 ? oneUpPlatformXMaxRight : oneUpPlatformXMaxLeft;
        float oneUpPlatformZMin = stickSol->xDir == 0 ? oneUpPlatformZMinRight : oneUpPlatformZMinLeft;
        float oneUpPlatformZMax = stickSol->xDir == 0 ? oneUpPlatformZMaxRight : oneUpPlatformZMaxLeft;

        float relY = stickSol->stickY + 6.0f;
        float intendedMag = (relY * relY / 128.0f);
        int intendedYaw = atan2sG(-relY, 0) + oupSol->cameraYaw;
        intendedYaw = (65536 + intendedYaw) % 65536;
        int intendedDYaw = (65536 + intendedYaw - oupSol->angle) % 65536;

        float lossFactor = gCosineTableG[intendedDYaw / 16];
        lossFactor *= 0.5f + 0.5f * stickSol->startSpeed / 100.0f;
        lossFactor = intendedMag / 32.0f * lossFactor * 0.02f + 0.92f;

        float startSpeedX = stickSol->startSpeed * gSineTableG[(oupSol->angle) / 16];
        float startSpeedZ = stickSol->startSpeed * gCosineTableG[(oupSol->angle) / 16];
        float newStartSpeed = sqrtf(startSpeedX * startSpeedX + startSpeedZ * startSpeedZ);

        float returnSpeedX = startSpeedX;
        float returnSpeedZ = startSpeedZ;

        returnSpeedX += returnSpeedZ * (intendedMag / 32.0f) * gSineTableG[intendedDYaw / 16] * 0.05f;
        returnSpeedZ -= returnSpeedX * (intendedMag / 32.0f) * gSineTableG[intendedDYaw / 16] * 0.05f;

        float newSpeed = sqrtf(returnSpeedX * returnSpeedX + returnSpeedZ * returnSpeedZ);

        returnSpeedX = returnSpeedX * newStartSpeed / newSpeed;
        returnSpeedZ = returnSpeedZ * newStartSpeed / newSpeed;

        returnSpeedX += 7.0f * oneUpPlatformNormalX;

        returnSpeedX *= lossFactor;
        returnSpeedZ *= lossFactor;

        float returnSpeed = -sqrtf(returnSpeedX * returnSpeedX + returnSpeedZ * returnSpeedZ);

        float minStartSpeed = INFINITY;
        float maxStartSpeed = -INFINITY;

        if (fabsf(returnSpeedX) > 0.0001) {
            double t = ((65536.0 * oupSol->pux) + oneUpPlatformXMin - platSol->returnPosition[0]) / -((oneUpPlatformNormalY + stickSol->q3 - 1.0f) * returnSpeedX / 4.0);
            float zCrossing = platSol->returnPosition[2] - (t * (oneUpPlatformNormalY + stickSol->q3 - 1.0f) * returnSpeedZ / 4.0f);

            if (zCrossing >= (65536.0 * oupSol->puz) + oneUpPlatformZMin && zCrossing <= (65536.0 * oupSol->puz) + oneUpPlatformZMax) {
                double p = (intendedMag * gCosineTableG[intendedDYaw / 16]);
                double eqB = (50.0 + 147200.0 / p);
                double eqC = -(320000.0 / p) * t * returnSpeed;
                double eqDet = eqB * eqB - eqC;

                if (eqDet > 0) {
                    float adjustedSpeed = sqrt(eqDet) - eqB;
                    minStartSpeed = fminf(minStartSpeed, adjustedSpeed);
                    maxStartSpeed = fmaxf(maxStartSpeed, adjustedSpeed);
                }
            }

            t = ((65536.0 * oupSol->pux) + oneUpPlatformXMax - platSol->returnPosition[0]) / -((oneUpPlatformNormalY + stickSol->q3 - 1.0f) * returnSpeedX / 4.0);
            zCrossing = platSol->returnPosition[2] - (t * (oneUpPlatformNormalY + stickSol->q3 - 1.0f) * returnSpeedZ / 4.0f);

            if (zCrossing >= (65536.0 * oupSol->puz) + oneUpPlatformZMin && zCrossing <= (65536.0 * oupSol->puz) + oneUpPlatformZMax) {
                double p = (intendedMag * gCosineTableG[intendedDYaw / 16]);
                double eqB = (50.0 + 147200.0 / p);
                double eqC = -(320000.0 / p) * t * returnSpeed;
                double eqDet = eqB * eqB - eqC;

                if (eqDet > 0) {
                    float adjustedSpeed = sqrt(eqDet) - eqB;
                    minStartSpeed = fminf(minStartSpeed, adjustedSpeed);
                    maxStartSpeed = fmaxf(maxStartSpeed, adjustedSpeed);
                }
            }
        }

        if (fabsf(returnSpeedZ) > 0.0001) {
            double t = ((65536.0 * oupSol->puz) + oneUpPlatformZMin - platSol->returnPosition[2]) / -((oneUpPlatformNormalY + stickSol->q3 - 1.0f) * returnSpeedZ / 4.0);
            float xCrossing = platSol->returnPosition[0] - (t * (oneUpPlatformNormalY + stickSol->q3 - 1.0f) * returnSpeedX / 4.0f);

            if (xCrossing >= (65536.0 * oupSol->pux) + oneUpPlatformXMin && xCrossing <= (65536.0 * oupSol->pux) + oneUpPlatformXMax) {
                double p = (intendedMag * gCosineTableG[intendedDYaw / 16]);
                double eqB = (50.0 + 147200.0 / p);
                double eqC = -(320000.0 / p) * t * returnSpeed;
                double eqDet = eqB * eqB - eqC;

                if (eqDet > 0) {
                    float adjustedSpeed = sqrt(eqDet) - eqB;
                    minStartSpeed = fminf(minStartSpeed, adjustedSpeed);
                    maxStartSpeed = fmaxf(maxStartSpeed, adjustedSpeed);
                }
            }

            t = ((65536.0 * oupSol->puz) + oneUpPlatformZMax - platSol->returnPosition[2]) / -((oneUpPlatformNormalY + stickSol->q3 - 1.0) * returnSpeedZ / 4.0);
            xCrossing = platSol->returnPosition[0] - (t * (oneUpPlatformNormalY + stickSol->q3 - 1.0) * returnSpeedX / 4.0f);

            if (xCrossing >= (65536.0 * oupSol->pux) + oneUpPlatformXMin && xCrossing <= (65536.0 * oupSol->pux) + oneUpPlatformXMax) {
                double p = (intendedMag * gCosineTableG[intendedDYaw / 16]);
                double eqB = (50.0 + 147200.0 / p);
                double eqC = -(320000.0 / p) * t * returnSpeed;
                double eqDet = eqB * eqB - eqC;

                if (eqDet > 0) {
                    float adjustedSpeed = sqrt(eqDet) - eqB;
                    minStartSpeed = fminf(minStartSpeed, adjustedSpeed);
                    maxStartSpeed = fmaxf(maxStartSpeed, adjustedSpeed);
                }
            }
        }

        if (minStartSpeed <= maxStartSpeed) {
            float minLossFactor = gCosineTableG[intendedDYaw / 16];
            minLossFactor *= 0.5f + 0.5f * minStartSpeed / 100.0f;
            minLossFactor = intendedMag / 32.0f * minLossFactor * 0.02f + 0.92f;

            float minStartSpeedX = minStartSpeed * gSineTableG[(oupSol->angle) / 16];
            float minStartSpeedZ = minStartSpeed * gCosineTableG[(oupSol->angle) / 16];

            float minReturnSpeedX = minStartSpeedX;
            float minReturnSpeedZ = minStartSpeedZ;

            minReturnSpeedX += minReturnSpeedZ * (intendedMag / 32.0f) * gSineTableG[intendedDYaw / 16] * 0.05f;
            minReturnSpeedZ -= minReturnSpeedX * (intendedMag / 32.0f) * gSineTableG[intendedDYaw / 16] * 0.05f;

            newSpeed = sqrtf(minReturnSpeedX * minReturnSpeedX + minReturnSpeedZ * minReturnSpeedZ);

            minReturnSpeedX = minReturnSpeedX * minStartSpeed / newSpeed;
            minReturnSpeedZ = minReturnSpeedZ * minStartSpeed / newSpeed;

            minReturnSpeedX += 7.0f * oneUpPlatformNormalX;

            minReturnSpeedX *= minLossFactor;
            minReturnSpeedZ *= minLossFactor;

            float minReturnSpeed = -sqrtf(minReturnSpeedX * minReturnSpeedX + minReturnSpeedZ * minReturnSpeedZ);

            float maxLossFactor = gCosineTableG[intendedDYaw / 16];
            maxLossFactor *= 0.5f + 0.5f * maxStartSpeed / 100.0f;
            maxLossFactor = intendedMag / 32.0f * maxLossFactor * 0.02f + 0.92f;

            float maxStartSpeedX = maxStartSpeed * gSineTableG[(oupSol->angle) / 16];
            float maxStartSpeedZ = maxStartSpeed * gCosineTableG[(oupSol->angle) / 16];

            float maxReturnSpeedX = maxStartSpeedX;
            float maxReturnSpeedZ = maxStartSpeedZ;

            maxReturnSpeedX += maxReturnSpeedZ * (intendedMag / 32.0f) * gSineTableG[intendedDYaw / 16] * 0.05f;
            maxReturnSpeedZ -= maxReturnSpeedX * (intendedMag / 32.0f) * gSineTableG[intendedDYaw / 16] * 0.05f;

            newSpeed = sqrtf(maxReturnSpeedX * maxReturnSpeedX + maxReturnSpeedZ * maxReturnSpeedZ);

            maxReturnSpeedX = maxReturnSpeedX * maxStartSpeed / newSpeed;
            maxReturnSpeedZ = maxReturnSpeedZ * maxStartSpeed / newSpeed;

            maxReturnSpeedX += 7.0f * oneUpPlatformNormalX;

            maxReturnSpeedX *= maxLossFactor;
            maxReturnSpeedZ *= maxLossFactor;

            float maxReturnSpeed = -sqrtf(maxReturnSpeedX * maxReturnSpeedX + maxReturnSpeedZ * maxReturnSpeedZ);

            double ax = platSol->returnPosition[0] - minReturnSpeedX * (oneUpPlatformNormalY + stickSol->q3 - 1.0f) / 4.0f - minStartSpeedX * (startNormals[stickSol->floorIdx][1] + stickSol->q1q2 - 1.0f) / 4.0;
            double az = platSol->returnPosition[2] - minReturnSpeedZ * (oneUpPlatformNormalY + stickSol->q3 - 1.0f) / 4.0f - minStartSpeedZ * (startNormals[stickSol->floorIdx][1] + stickSol->q1q2 - 1.0f) / 4.0;
            double bx = platSol->returnPosition[0] - maxReturnSpeedX * (oneUpPlatformNormalY + stickSol->q3 - 1.0f) / 4.0f - maxStartSpeedX * (startNormals[stickSol->floorIdx][1] + stickSol->q1q2 - 1.0f) / 4.0;
            double bz = platSol->returnPosition[2] - maxReturnSpeedZ * (oneUpPlatformNormalY + stickSol->q3 - 1.0f) / 4.0f - maxStartSpeedZ * (startNormals[stickSol->floorIdx][1] + stickSol->q1q2 - 1.0f) / 4.0;

            for (int i = 0; i < 3; i++) {
                if ((stickSol->floorIdx == 0 && ((i == 0 && squishCeilings[2]) || (i == 1 && squishCeilings[0]))) || (stickSol->floorIdx == 1 && ((i == 1 && squishCeilings[1]) || (i == 2 && squishCeilings[3])))) {
                    double px = startTriangles[stickSol->floorIdx][i][0];
                    double pz = startTriangles[stickSol->floorIdx][i][2];
                    double qx = startTriangles[stickSol->floorIdx][(i + 1) % 3][0];
                    double qz = startTriangles[stickSol->floorIdx][(i + 1) % 3][2];

                    double t = ((qx - px) * (az - pz) - (qz - pz) * (ax - px)) / ((qz - pz) * (bx - ax) - (qx - px) * (bz - az));
                    double s = ((bx - ax) * t + (ax - px)) / (qx - px);

                    if (t >= 0 && t <= 1 && s >= 0 && s <= 1) {
                        int speedSolIdx = atomicAdd(&nSpeedSolutionsHAU, 1);

                        if (speedSolIdx < MAX_SPEED_SOLUTIONS_HAU) {
                            float trueStartSpeed = minStartSpeed + t * (maxStartSpeed - minStartSpeed);

                            SpeedSolutionHAU sol;
                            sol.oupSolutionIdx = idx;
                            sol.startSpeed = trueStartSpeed;
                            speedSolutionsHAU[speedSolIdx] = sol;
                        }
                    }
                }
            }
        }
    }
}

__global__ void check_speed_angle() {
    long long int idx = (long long int)blockIdx.x * (long long int)blockDim.x + (long long int)threadIdx.x;

    if (idx < (long long int)min(nStickSolutions, MAX_STICK_SOLUTIONS) * 2048) {
        float cameraPositions[4][3] = { {-8192, -2918, -8192}, {-8192, -2918, 8191}, {8191, -2918, -8192}, {8191, -2918, 8191} };
        float oupBuffer = 1000.0f;

        int solIdx = idx / 2048;

        struct StickSolution* stickSol = &(stickSolutions[solIdx]);
        struct UpwarpSolution* uwSol = &(upwarpSolutions[stickSol->upwarpSolutionIdx]);
        struct PlatformSolution* platSol = &(platSolutions[uwSol->platformSolutionIdx]);

        int hau = (idx % 2048) + (stickSol->xDir == 0 ? 2048 : 0);

        float oneUpPlatformNormalX = stickSol->xDir == 0 ? oneUpPlatformNormalXRight : oneUpPlatformNormalXLeft;
        float oneUpPlatformNormalY = stickSol->xDir == 0 ? oneUpPlatformNormalYRight : oneUpPlatformNormalYLeft;
        float oneUpPlatformXMin = stickSol->xDir == 0 ? oneUpPlatformXMinRight : oneUpPlatformXMinLeft;
        float oneUpPlatformXMax = stickSol->xDir == 0 ? oneUpPlatformXMaxRight : oneUpPlatformXMaxLeft;
        float oneUpPlatformZMin = stickSol->xDir == 0 ? oneUpPlatformZMinRight : oneUpPlatformZMinLeft;
        float oneUpPlatformZMax = stickSol->xDir == 0 ? oneUpPlatformZMaxRight : oneUpPlatformZMaxLeft;

        float relY = stickSol->stickY + 6.0f;
        float intendedMag = (relY * relY / 128.0f);
        float tenKMul = -(0.5f + 0.5f * stickSol->startSpeed / 100.0f);
        tenKMul = intendedMag / 32.0f * tenKMul * 0.02f + 0.92f;
        float returnSpeed = stickSol->startSpeed * tenKMul;
        float returnSpeedX = returnSpeed * gSineTableG[hau];
        float returnSpeedZ = returnSpeed * gCosineTableG[hau];

        float startSpeedX = stickSol->startSpeed * gSineTableG[hau];
        float startSpeedZ = stickSol->startSpeed * gCosineTableG[hau];

        float oupX = platSol->returnPosition[0] - (oneUpPlatformNormalY + stickSol->q3 - 1.0f) * (returnSpeedX / 4.0f);
        float oupZ = platSol->returnPosition[2] - (oneUpPlatformNormalY + stickSol->q3 - 1.0f) * (returnSpeedZ / 4.0f);

        if (oupX >= INT32_MIN && oupX <= INT32_MAX && oupZ >= INT32_MIN && oupZ <= INT32_MAX) {
            short relX = (short)(int)oupX;
            short relZ = (short)(int)oupZ;

            if (relX >= oneUpPlatformXMin - oupBuffer && relX <= oneUpPlatformXMax + oupBuffer && relZ >= oneUpPlatformZMin - oupBuffer && relZ <= oneUpPlatformZMax + oupBuffer) {
                bool inBoundsTest = true;
                int oobTolerance = 1000;

                float intendedPosition[3];
                intendedPosition[0] = oupX;
                intendedPosition[2] = oupZ;

                float currentNormalY = oneUpPlatformNormalY;

                for (int j = 1; j <= stickSol->q3; j++) {
                    intendedPosition[0] = intendedPosition[0] + currentNormalY * returnSpeedX / 4.0f;
                    intendedPosition[2] = intendedPosition[2] + currentNormalY * returnSpeedZ / 4.0f;
                    currentNormalY = 1.0f;

                    if ((short)(int)intendedPosition[0] < -8191 - oobTolerance || (short)(int)intendedPosition[0] > 8192 + oobTolerance || (short)(int)intendedPosition[2] < -8191 - oobTolerance || (short)(int)intendedPosition[2] > 8192 + oobTolerance) {
                        inBoundsTest = false;
                        break;
                    }
                }

                if (inBoundsTest) {
                    intendedPosition[0] = oupX - (startNormals[stickSol->floorIdx][1] + stickSol->q1q2 - 1.0f) * (startSpeedX / 4.0f);
                    intendedPosition[2] = oupZ - (startNormals[stickSol->floorIdx][1] + stickSol->q1q2 - 1.0f) * (startSpeedZ / 4.0f);
                    float currentNormalY = startNormals[stickSol->floorIdx][1];

                    for (int j = 1; j <= stickSol->q1q2; j++) {
                        intendedPosition[0] = intendedPosition[0] + currentNormalY * startSpeedX / 4.0f;
                        intendedPosition[2] = intendedPosition[2] + currentNormalY * startSpeedZ / 4.0f;
                        currentNormalY = 1.0f;

                        if ((short)(int)intendedPosition[0] < -8191 - oobTolerance || (short)(int)intendedPosition[0] > 8192 + oobTolerance || (short)(int)intendedPosition[2] < -8191 - oobTolerance || (short)(int)intendedPosition[2] > 8192 + oobTolerance) {
                            inBoundsTest = false;
                            break;
                        }
                    }
                }

                if (inBoundsTest) {
                    float startSpeedX = stickSol->startSpeed * gSineTableG[hau];
                    float startSpeedZ = stickSol->startSpeed * gCosineTableG[hau];

                    int minCameraYaw = 0;
                    int maxCameraYaw = 0;

                    float oneUpPlatformPosition[3] = { oupX, platSol->returnPosition[1], oupZ };

                    int refCameraYaw = calculate_camera_yaw(oneUpPlatformPosition, cameraPositions[0]);
                    refCameraYaw = (65536 + refCameraYaw) % 65536;

                    for (int k = 1; k < 4; k++) {
                        int cameraYaw = calculate_camera_yaw(oneUpPlatformPosition, cameraPositions[k]);
                        cameraYaw = (short)(cameraYaw - refCameraYaw);
                        minCameraYaw = min(minCameraYaw, cameraYaw);
                        maxCameraYaw = max(maxCameraYaw, cameraYaw);
                    }

                    int minCameraIdx = gReverseArctanTable[(65536 + minCameraYaw + refCameraYaw) % 65536];
                    int maxCameraIdx = gReverseArctanTable[(65536 + maxCameraYaw + refCameraYaw) % 65536];

                    if (minCameraIdx > maxCameraIdx) {
                        maxCameraIdx += 8192;
                    }

                    for (int cIdx = minCameraIdx; cIdx <= maxCameraIdx; cIdx++) {
                        int cameraYaw = gArctanTableG[(8192 + cIdx) % 8192];
                        cameraYaw = (65536 + cameraYaw) % 65536;

                        if (validCameraAngle[cameraYaw]) {
                            int oupSolIdx = atomicAdd(&nOUPSolutions, 1);

                            if (oupSolIdx < MAX_OUP_SOLUTIONS) {
                                OUPSolution sol;
                                sol.stickSolutionIdx = solIdx;
                                sol.pux = (int)floor((oupX + 32768.0) / 65536.0);
                                sol.puz = (int)floor((oupZ + 32768.0) / 65536.0);
                                sol.angle = 16 * hau;
                                sol.cameraYaw = cameraYaw;
                                oupSolutions[oupSolIdx] = sol;
                            }
                        }
                    }
                }
            }
        }
    }
}

__global__ void find_stick_solutions() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nUpwarpSolutions, MAX_UPWARP_SOLUTIONS)) {
        double speedBuffer = 5000.0;

        struct UpwarpSolution* uwSol = &(upwarpSolutions[idx]);
        struct PlatformSolution* platSol = &(platSolutions[uwSol->platformSolutionIdx]);
        float floorHeight;
        int floorIdx = find_floor_triangles(platSol->endPosition, platSol->endTriangles, platSol->endTriangleNormals, &floorHeight);

        for (int i = 0; i < 112; i++) {
            double mul = tenKMultipliers[i];

            if (!isnan(mul)) {
                double slideSpeed = -65536.0 * sqrt((double)uwSol->pux * (double)uwSol->pux + (double)uwSol->puz * (double)uwSol->puz) / (double)platSol->endTriangleNormals[floorIdx][1];
                double minPre10KSpeed = (slideSpeed / 0.94) / mul - speedBuffer;
                double maxPre10KSpeed = (slideSpeed / 0.9) / mul + speedBuffer;

                double minStickMag = sqrt(-128 * (320000.0 * mul - 294400.0) / (100.0 + maxPre10KSpeed));
                double maxStickMag = sqrt(-128 * (320000.0 * mul - 294400.0) / (100.0 + minPre10KSpeed));

                int minJ = max(2, (int)ceil(minStickMag));
                int maxJ = min(32, (int)floor(maxStickMag));

                for (int j = minJ; j <= maxJ; j++) {
                    int solIdx = atomicAdd(&nStickSolutions, 1);

                    if (solIdx < MAX_STICK_SOLUTIONS) {
                        float speed = -((-320000.0 * mul + 294400.0) / -((double)(j * j) / 128.0) + 100.0);

                        StickSolution solution;
                        solution.upwarpSolutionIdx = idx;
                        solution.stickY = -j - 6;
                        solution.startSpeed = speed;
                        int temp = i;
                        solution.q1q2 = (temp % 7) + 2;
                        temp = temp / 7;
                        solution.q3 = (temp % 4) + 1;
                        temp = temp / 4;
                        solution.floorIdx = temp % 2;
                        temp = temp / 2;
                        solution.xDir = temp;
                        stickSolutions[solIdx] = solution;
                    }
                }
            }
        }
    }
}

void run_hau_bruteforcer(int g, int h, int i, int j, float normX, float normY, float normZ, float* host_norms, std::ofstream& wf, char* normalStages, float* finalHeightDiffs)
{
    int current_normal_index = ((g * nSamplesNY + h) * nSamplesNX + i) * nSamplesNZ + j;

    cudaMemcpyToSymbol(nStickSolutions, &nStickSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

    find_stick_solutions << <nBlocks, nThreads >> > ();

    cudaMemcpyFromSymbol(&nStickSolutionsCPU, nStickSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);

    if (nStickSolutionsCPU > 0) {
        normalStages[current_normal_index] = 3;

        if (nStickSolutionsCPU > MAX_STICK_SOLUTIONS) {
            fprintf(stderr, "Warning: Number of stick solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
            nStickSolutionsCPU = MAX_STICK_SOLUTIONS;
        }

        if (subSolutionPrintingMode == 2)
            printf("  Stage 3 Solutions: %d\n", nStickSolutionsCPU);

        cudaMemcpyToSymbol(nOUPSolutions, &nOUPSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

        /* UNCOMMENT TO CHECK STICK SOLUTIONS
        *
        * nBlocks = (nStickSolutionsCPU + nThreads - 1) / nThreads;
        *
        * check_stick_solutions_for_the_right_one << < nBlocks, nThreads >> > ();
        */

        nBlocks = ((long long)2048 * (long long)nStickSolutionsCPU + (long long)nThreads - (long long)1) / (long long)nThreads;

        check_speed_angle << <nBlocks, nThreads >> > ();

        cudaMemcpyFromSymbol(&nOUPSolutionsCPU, nOUPSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
    }
    else
    {
        if (subSolutionPrintingMode == 2)
            printf("  Stage 3 Solutions: Failed\n");
    }

    if (nOUPSolutionsCPU > 0) {
        normalStages[current_normal_index] = 4;

        if (nOUPSolutionsCPU > MAX_OUP_SOLUTIONS) {
            fprintf(stderr, "Warning: Number of 1-up Platform solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
            nOUPSolutionsCPU = MAX_OUP_SOLUTIONS;
        }

        if (subSolutionPrintingMode == 2)
            printf("  Stage 4 Solutions: %d\n", nOUPSolutionsCPU);

        cudaMemcpyToSymbol(nSpeedSolutionsHAU, &nSpeedSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

        nBlocks = (nOUPSolutionsCPU + nThreads - 1) / nThreads;

        test_oup_solution << <nBlocks, nThreads >> > ();

        cudaMemcpyFromSymbol(&nSpeedSolutionsCPU, nSpeedSolutionsHAU, sizeof(int), 0, cudaMemcpyDeviceToHost);
    }
    else
    {
        if (subSolutionPrintingMode == 2)
            printf("  Stage 4 Solutions: Failed\n");
    }

    if (nSpeedSolutionsCPU > 0) {
        normalStages[current_normal_index] = 5;

        if (nSpeedSolutionsCPU > MAX_SPEED_SOLUTIONS_HAU) {
            fprintf(stderr, "Warning: Number of speed solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
            nSpeedSolutionsCPU = MAX_SPEED_SOLUTIONS_HAU;

        }

        if (subSolutionPrintingMode == 1)
            printf("# Speed Solutions (%d, %d, %d, %d): %d\n\n", g, h, i, j, nSpeedSolutionsCPU);
        if (subSolutionPrintingMode == 2)
            printf("  Stage 5 Solutions: %d\n", nSpeedSolutionsCPU);


        cudaMemcpyToSymbol(n10KSolutionsHAU, &n10KSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

        // TODO - REMOVE
        int nPass1SolsCPU = 0;
        int nPass2SolsCPU = 0;
        int nPass3SolsCPU = 0;
        float currentLowestHeightDiffCPU = MAX_HEIGHT_DIFFERENCE;

        cudaMemcpyToSymbol(nPass1Sols, &nPass1SolsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(nPass2Sols, &nPass2SolsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(nPass3Sols, &nPass3SolsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

        cudaMemcpyToSymbol(currentLowestHeightDiff, &currentLowestHeightDiffCPU, sizeof(float), 0, cudaMemcpyHostToDevice);

        nBlocks = (nSpeedSolutionsCPU + nThreads - 1) / nThreads;

        test_speed_solution << <nBlocks, nThreads >> > ();

        cudaMemcpyFromSymbol(&n10KSolutionsCPU, n10KSolutionsHAU, sizeof(int), 0, cudaMemcpyDeviceToHost);

        cudaMemcpyFromSymbol(&nPass1SolsCPU, nPass1Sols, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&nPass2SolsCPU, nPass2Sols, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&nPass3SolsCPU, nPass3Sols, sizeof(int), 0, cudaMemcpyDeviceToHost);

        cudaMemcpyFromSymbol(&currentLowestHeightDiffCPU, currentLowestHeightDiff, sizeof(float), 0, cudaMemcpyDeviceToHost);

        if (nPass1SolsCPU > 0)
        {
            normalStages[current_normal_index]++;
            if (subSolutionPrintingMode == 2)
                printf("  Stage 6 Pass Count: %d\n", nPass1SolsCPU);
        }
        else
        {
            if (subSolutionPrintingMode == 2)
                printf("  Stage 6 Pass Count: Failed\n");
        }
        if (nPass2SolsCPU > 0)
        {
            normalStages[current_normal_index]++;
            if (subSolutionPrintingMode == 2)
                printf("  Stage 7 Pass Count: %d\n", nPass2SolsCPU);
        }
        else
        {
            if (subSolutionPrintingMode == 2)
                printf("  Stage 7 Pass Count: Failed\n");
        }
        if (nPass3SolsCPU > 0)
        {
            normalStages[current_normal_index]++;
            if (subSolutionPrintingMode == 2)
                printf("  Stage 8 Pass Count: %d\n", nPass3SolsCPU);
        }
        else
        {
            if (subSolutionPrintingMode == 2)
                printf("  Stage 8 Pass Count: Failed\n");
        }

        finalHeightDiffs[current_normal_index] = currentLowestHeightDiffCPU;
    }
    else
    {
        if (subSolutionPrintingMode == 2)
        {
            printf("  Stage 5 Solutions: Failed\n");
            printf("  Stage 6 Pass Count: Failed\n");
            printf("  Stage 7 Pass Count: Failed\n");
            printf("  Stage 8 Pass Count: Failed\n");
        }
    }

    if (n10KSolutionsCPU > 0) {
        normalStages[((g * nSamplesNY + h) * nSamplesNX + i) * nSamplesNZ + j] = 9; //TODO - SWITCH BACK TO 6

        if (n10KSolutionsCPU > MAX_10K_SOLUTIONS_HAU) {
            fprintf(stderr, "Warning: Number of 10k solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
            n10KSolutionsCPU = MAX_10K_SOLUTIONS_HAU;
        }


        if (subSolutionPrintingMode == 1)
            printf("# 10k Solutions (%d, %d, %d, %d): %d\n\n", g, h, i, j, n10KSolutionsCPU);
        if (subSolutionPrintingMode == 2)
            printf("  Stage 9 Solutions: %d\n", nSpeedSolutionsCPU);

        struct PlatformSolution* platSolutionsCPU = (struct PlatformSolution*)std::malloc(nPlatSolutionsCPU * sizeof(struct PlatformSolution));
        struct UpwarpSolution* upwarpSolutionsCPU = (struct UpwarpSolution*)std::malloc(nUpwarpSolutionsCPU * sizeof(struct UpwarpSolution));
        struct StickSolution* stickSolutionsCPU = (struct StickSolution*)std::malloc(nStickSolutionsCPU * sizeof(struct StickSolution));
        struct OUPSolution* oupSolutionsCPU = (struct OUPSolution*)std::malloc(nOUPSolutionsCPU * sizeof(struct OUPSolution));
        struct SpeedSolutionHAU* speedSolutionsCPU = (struct SpeedSolutionHAU*)std::malloc(nSpeedSolutionsCPU * sizeof(struct SpeedSolutionHAU));
        struct TenKSolutionHAU* tenKSolutionsCPU = (struct TenKSolutionHAU*)std::malloc(n10KSolutionsCPU * sizeof(struct TenKSolutionHAU));

        cudaMemcpyFromSymbol(tenKSolutionsCPU, tenKSolutionsHAU, n10KSolutionsCPU * sizeof(struct TenKSolutionHAU), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(speedSolutionsCPU, speedSolutionsHAU, nSpeedSolutionsCPU * sizeof(struct SpeedSolutionHAU), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(oupSolutionsCPU, oupSolutions, nOUPSolutionsCPU * sizeof(struct OUPSolution), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(stickSolutionsCPU, stickSolutions, nStickSolutionsCPU * sizeof(struct StickSolution), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(upwarpSolutionsCPU, upwarpSolutions, nUpwarpSolutionsCPU * sizeof(struct UpwarpSolution), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(platSolutionsCPU, platSolutions, nPlatSolutionsCPU * sizeof(struct PlatformSolution), 0, cudaMemcpyDeviceToHost);

        for (int l = 0; l < n10KSolutionsCPU; l++) {
            struct TenKSolutionHAU* tenKSol = &(tenKSolutionsCPU[l]);
            struct SpeedSolutionHAU* speedSol = &(speedSolutionsCPU[tenKSol->speedSolutionIdx]);
            struct OUPSolution* oupSol = &(oupSolutionsCPU[speedSol->oupSolutionIdx]);
            struct StickSolution* stickSol = &(stickSolutionsCPU[oupSol->stickSolutionIdx]);
            struct UpwarpSolution* uwSol = &(upwarpSolutionsCPU[stickSol->upwarpSolutionIdx]);
            struct PlatformSolution* platSol = &(platSolutionsCPU[uwSol->platformSolutionIdx]);

            wf << normX << ", " << normY << ", " << normZ << ", ";
            wf << tenKSol->startPosition[0] << ", " << tenKSol->startPosition[1] << ", " << tenKSol->startPosition[2] << ", ";
            wf << tenKSol->frame1Position[0] << ", " << tenKSol->frame1Position[1] << ", " << tenKSol->frame1Position[2] << ", ";
            wf << tenKSol->frame2Position[0] << ", " << tenKSol->frame2Position[1] << ", " << tenKSol->frame2Position[2] << ", ";
            wf << platSol->returnPosition[0] << ", " << platSol->returnPosition[1] << ", " << platSol->returnPosition[2] << ", ";
            wf << speedSol->startSpeed << ", " << tenKSol->startSpeedX << ", " << tenKSol->startSpeedZ << ", ";
            wf << tenKSol->returnSpeed << ", " << tenKSol->returnSpeedX << ", " << tenKSol->returnSpeedZ << ", ";
            wf << tenKSol->frame1QSteps << ", " << tenKSol->frame2QSteps << ", " << stickSol->q3 << ", ";
            wf << oupSol->angle << ", ";
            wf << 0 << ", " << stickSol->stickY << ", ";
            wf << oupSol->cameraYaw << ", ";
            wf << host_norms[3 * stickSol->floorIdx] << ", " << host_norms[3 * stickSol->floorIdx + 1] << ", " << host_norms[3 * stickSol->floorIdx + 2] << ", ";
            wf << platSol->nFrames << ", ";
            wf << platSol->endNormal[0] << ", " << platSol->endNormal[1] << ", " << platSol->endNormal[2] << ", ";
            wf << platSol->endPosition[0] << ", " << platSol->endPosition[1] << ", " << platSol->endPosition[2] << ", ";
            wf << uwSol->upwarpPosition[0] << ", " << uwSol->upwarpPosition[1] << ", " << uwSol->upwarpPosition[2] << ", ";
            wf << uwSol->pux << ", " << uwSol->puz << std::endl;
        }

        free(platSolutionsCPU);
        free(upwarpSolutionsCPU);
        free(stickSolutionsCPU);
        free(oupSolutionsCPU);
        free(speedSolutionsCPU);
        free(tenKSolutionsCPU);
    }
    else
    {
        if (subSolutionPrintingMode == 2)
            printf("  Stage 9 Solutions: Failed\n");
    }
}


// Checks to see if a PU solution with the correct parameters was found.
__global__ void check_stick_solutions_for_the_right_one()
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nStickSolutions, MAX_STICK_SOLUTIONS)) {
        struct StickSolution* stickSol = &(stickSolutions[idx]);

        if (stickSol->stickY == -11 && stickSol->startSpeed == 10226649 && abs(startNormals[stickSol->floorIdx][0] + 0.3858663738) < 0.0001 && abs(startNormals[stickSol->floorIdx][1] - 0.8713111877) < 0.0001 && abs(startNormals[stickSol->floorIdx][2] - 0.3031898439) < 0.0001 && stickSol->q3 == 1)
        {
            struct UpwarpSolution* upSol = &(upwarpSolutions[stickSol->upwarpSolutionIdx]);

            if (upSol->pux == 116 && upSol->puz == -640 && abs(upSol->upwarpPosition[0] - 7413681.5) < 0.1 && abs(upSol->upwarpPosition[1] + 1177.262451) < 0.1 && abs(upSol->upwarpPosition[2] + 42003140) < 10)
            {
                struct PlatformSolution* platSol = &(platSolutions[upSol->platformSolutionIdx]);

                //printf("Matching Upwarp Solution Found!\n Index: %i\n PlatSolEndNormal: %f, %f, %f\n\n", idx, platSol->endNormal[0], platSol->endNormal[1], platSol->endNormal[2]);

                if (abs(platSol->endNormal[0] - 0.1899999082) < 0.0001 && abs(platSol->endNormal[1] - 0.843072772) < 0.0001 && abs(platSol->endNormal[2] - 0.4891075492) < 0.0001 &&
                    abs(platSol->endPosition[0] + 1812.662598) < 0.0001 && abs(platSol->endPosition[1] + 3065.262451) < 0.0001 && abs(platSol->endPosition[2] + 424.9223938) < 0.0001 &&
                    abs(platSol->returnPosition[0] + 1864.5) < 0.0001 && abs(platSol->returnPosition[1] + 2928.717285) < 0.0001 && abs(platSol->returnPosition[2] + 454) < 0.0001 &&
                    platSol->nFrames == 55)
                {
                    printf("Correct Stick Solution found! Index: %i\n", idx);
                    printf("%i Floor Index: %d\n", idx, stickSol->floorIdx);

                    if (!stickSol->floorIdx)
                    {
                        printf("Setting correctStickSolIdx now!\n");
                        correctStickSolIdx = idx;
                    }
                    else
                    {
                        printf("Setting correctStickSolIdx1 now!\n");
                        correctStickSolIdx1 = idx;
                    }
                }
            }
        }
    }
}

void setup_output_hau(std::ofstream& wf)
{
    wf << std::fixed;

    wf << "Start Normal X, Start Normal Y, Start Normal Z, ";
    wf << "Start Position X, Start Position Y, Start Position Z, ";
    wf << "Frame 1 Position X, Frame 1 Position Y, Frame 1 Position Z, ";
    wf << "1-up Platform Position X, 1-up Platform Position Y, 1-up Platform Position Z, ";
    wf << "Return Position X, Return Position Y, Return Position Z, ";
    wf << "Pre-10K Speed, Pre-10K X Velocity, Pre-10K Z Velocity, ";
    wf << "Return Speed, Return X Velocity, Return Z Velocity, ";
    wf << "Frame 1 Q-steps, Frame 2 Q-steps, Frame 3 Q-steps, ";
    wf << "Frame 1 Angle, ";
    wf << "10K Stick X, 10K Stick Y, ";
    wf << "10K Camera Yaw, ";
    wf << "Start Floor Normal X, Start Floor Normal Y, Start Floor Normal Z, ";
    wf << "Number of Tilt Frames, ";
    wf << "Post-Tilt Platform Normal X, Post-Tilt Platform Normal Y, Post-Tilt Platform Normal Z, ";
    wf << "Post-Tilt Position X, Post-Tilt Position Y, Post-Tilt Position Z, ";
    wf << "Post-Upwarp Position X, Post-Upwarp Position Y, Post-Upwarp Position Z, ";
    wf << "Upwarp PU X, Upwarp PU Z";
    wf << std::endl;
}