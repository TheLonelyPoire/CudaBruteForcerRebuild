#include "SlideKickFunctions.cuh"

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


void setup_output_slide_kick(std::ofstream& wf)
{
    wf << std::fixed;

    wf << "Start Normal X, Start Normal Y, Start Normal Z, ";
    wf << "Start Position X, Start Position Y, Start Position Z, ";
    wf << "Frame 1 Position X, Frame 1 Position Y, Frame 1 Position Z, ";
    wf << "Frame 2 Position X, Frame 2 Position Y, Frame 2 Position Z, ";
    wf << "Return Position X, Return Position Y, Return Position Z, ";
    wf << "Departure Speed, Pre-10K X Velocity, Pre-10K Z Velocity, ";
    wf << "Return Speed, Return X Velocity, Return Z Velocity, ";
    wf << "Frame 1 Q-steps, Frame 2 Q-steps, Frame 3 Q-steps, ";
    wf << "10K Stick X, 10K Stick Y, ";
    wf << "Frame 2 HAU, 10K Camera Yaw, ";
    wf << "Start Floor Normal X, Start Floor Normal Y, Start Floor Normal Z, ";
    wf << "Number of Tilt Frames, ";
    wf << "Post-Tilt Platform Normal X, Post-Tilt Platform Normal Y, Post-Tilt Platform Normal Z, ";
    wf << "Post-Tilt Position X, Post-Tilt Position Y, Post-Tilt Position Z, ";
    wf << "Pre-Upwarp Position X, Pre-Upwarp Position Y, Pre-Upwarp Position Z, ";
    wf << "Post-Upwarp Position X, Post-Upwarp Position Y, Post-Upwarp Position Z, ";
    wf << "Upwarp PU X, Upwarp PU Z, ";
    wf << "Upwarp Slide Facing Angle, Upwarp Slide IntendedMag, Upwarp Slide IntendedDYaw, ";
    wf << "Post-Breakdance Camera Yaw, ";
    wf << "Post-Breakdance Stick X, Post-Breakdance Stick Y, ";
    wf << "Landing Position X, Landing Position Y, Landing Position Z, ";
    wf << "Landing Speed" << std::endl;
}

__global__ void copy_solution_pointers_sk(SKSolStruct s) {
    sk1Solutions = s.sk1Solutions;
    sk2ASolutions = s.sk2ASolutions;
    sk2BSolutions = s.sk2BSolutions;
    sk2CSolutions = s.sk2CSolutions;
    sk2DSolutions = s.sk2DSolutions;
    sk3Solutions = s.sk3Solutions;
    sk4Solutions = s.sk4Solutions;
    sk5Solutions = s.sk5Solutions;
    sk6Solutions = s.sk6Solutions;
    platSolutions = s.platSolutions;
    upwarpSolutions = s.upwarpSolutions;
    skuwSolutions = s.skuwSolutions;
    speedSolutionsSK = s.speedSolutions;
    tenKSolutionsSK = s.tenKSolutions;
    slideSolutions = s.slideSolutions;
    bdSolutions = s.bdSolutions;
}

void init_solution_structs_sk(SKSolStruct* s) {
    cudaMalloc((void**)&s->sk1Solutions, MAX_SK_PHASE_ONE * sizeof(SKPhase1));
    cudaMalloc((void**)&s->sk2ASolutions, MAX_SK_PHASE_TWO_A * sizeof(SKPhase2));
    cudaMalloc((void**)&s->sk2BSolutions, MAX_SK_PHASE_TWO_B * sizeof(SKPhase2));
    cudaMalloc((void**)&s->sk2CSolutions, MAX_SK_PHASE_TWO_C * sizeof(SKPhase2));
    cudaMalloc((void**)&s->sk2DSolutions, MAX_SK_PHASE_TWO_D * sizeof(SKPhase2));
    cudaMalloc((void**)&s->sk3Solutions, MAX_SK_PHASE_THREE * sizeof(SKPhase3));
    cudaMalloc((void**)&s->sk4Solutions, MAX_SK_PHASE_FOUR * sizeof(SKPhase4));
    cudaMalloc((void**)&s->sk5Solutions, MAX_SK_PHASE_FIVE * sizeof(SKPhase5));
    cudaMalloc((void**)&s->sk6Solutions, MAX_SK_PHASE_SIX * sizeof(SKPhase6));
    cudaMalloc((void**)&s->platSolutions, MAX_PLAT_SOLUTIONS * sizeof(PlatformSolution));
    cudaMalloc((void**)&s->upwarpSolutions, MAX_UPWARP_SOLUTIONS * sizeof(UpwarpSolution));
    cudaMalloc((void**)&s->skuwSolutions, MAX_SK_UPWARP_SOLUTIONS * sizeof(SKUpwarpSolution));
    cudaMalloc((void**)&s->speedSolutions, MAX_SPEED_SOLUTIONS_SK * sizeof(SpeedSolutionSK));
    cudaMalloc((void**)&s->tenKSolutions, MAX_10K_SOLUTIONS_SK * sizeof(TenKSolutionSK));
    cudaMalloc((void**)&s->slideSolutions, MAX_SLIDE_SOLUTIONS * sizeof(SlideSolution));
    cudaMalloc((void**)&s->bdSolutions, MAX_BD_SOLUTIONS * sizeof(BDSolution));

    copy_solution_pointers_sk << <1, 1 >> > (*s);
}

void free_solution_pointers_sk(SKSolStruct* s) {
    cudaFree(s->sk1Solutions);
    cudaFree(s->sk2ASolutions);
    cudaFree(s->sk2BSolutions);
    cudaFree(s->sk2CSolutions);
    cudaFree(s->sk2DSolutions);
    cudaFree(s->sk3Solutions);
    cudaFree(s->sk4Solutions);
    cudaFree(s->sk5Solutions);
    cudaFree(s->sk6Solutions);
    cudaFree(s->platSolutions);
    cudaFree(s->upwarpSolutions);
    cudaFree(s->skuwSolutions);
    cudaFree(s->speedSolutions);
    cudaFree(s->tenKSolutions);
    cudaFree(s->slideSolutions);
}

// Phase Functions

__global__ void try_stick_positionG() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nSK5Solutions, MAX_SK_PHASE_FIVE)) {
        struct SKPhase5* sol5 = &(sk5Solutions[idx]);
        struct SKPhase4* sol4 = &(sk4Solutions[sol5->p4Idx]);
        struct SKPhase3* sol3 = &(sk3Solutions[sol4->p3Idx]);
        struct SKPhase2* sol2 = (sol3->p2Type / 2 == 0) ? ((sol3->p2Type % 2 == 0) ? &(sk2ASolutions[sol3->p2Idx]) : &(sk2BSolutions[sol3->p2Idx])) : ((sol3->p2Type % 2 == 0) ? &(sk2CSolutions[sol3->p2Idx]) : &(sk2DSolutions[sol3->p2Idx]));
        struct SKPhase1* sol1 = &(sk1Solutions[sol2->p1Idx]);

        int trueX = (sol5->stickX == 0) ? 0 : ((sol5->stickX < 0) ? sol5->stickX - 6 : sol5->stickX + 6);
        int trueY = (sol5->stickY == 0) ? 0 : ((sol5->stickY < 0) ? sol5->stickY - 6 : sol5->stickY + 6);

        float mag = sqrtf((float)(sol5->stickX * sol5->stickX + sol5->stickY * sol5->stickY));

        float xS = sol5->stickX;
        float yS = sol5->stickY;

        if (mag > 64.0f) {
            xS = xS * (64.0f / mag);
            yS = yS * (64.0f / mag);
            mag = 64.0f;
        }

        float intendedMag = ((mag / 64.0f) * (mag / 64.0f)) * 32.0f;
        int intendedYaw = atan2sG(-yS, xS) + sol4->cameraYaw;
        int intendedDYaw = intendedYaw - sol5->f1Angle;
        intendedDYaw = (65536 + (intendedDYaw % 65536)) % 65536;

        float lower10KSpeed = sol4->minPre10KSpeed;
        float upper10KSpeed = sol4->maxPre10KSpeed;

        float forward = gCosineTableG[intendedDYaw >> 4];
        forward *= 0.5f + 0.5f * lower10KSpeed / 100.0f;
        float lossFactor = intendedMag / 32.0f * forward * 0.02f + 0.92f;

        lower10KSpeed *= lossFactor;
        forward = gCosineTableG[intendedDYaw >> 4];
        forward *= 0.5f + 0.5f * upper10KSpeed / 100.0f;
        lossFactor = intendedMag / 32.0f * forward * 0.02f + 0.92f;

        upper10KSpeed *= lossFactor;

        lower10KSpeed = fminf(sol4->minPost10KSpeed, lower10KSpeed);
        upper10KSpeed = fmaxf(sol4->maxPost10KSpeed, upper10KSpeed);

        if (lower10KSpeed >= upper10KSpeed) {
            float xVel = gSineTableG[sol2->f2Angle >> 4];
            float zVel = gCosineTableG[sol2->f2Angle >> 4];

            xVel += zVel * (intendedMag / 32.0f) * gSineTableG[intendedDYaw >> 4] * 0.05f;
            zVel -= xVel * (intendedMag / 32.0f) * gSineTableG[intendedDYaw >> 4] * 0.05f;

            double f3Angle = 65536.0 * atan2(xVel, zVel) / (2.0 * M_PI);

            double angleDiff = fmod(65536.0 + f3Angle - sol2->f2Angle, 65536.0);
            angleDiff = fmod(angleDiff + 32768.0, 65536.0) - 32768.0;

            if (angleDiff >= sol4->minAngleDiff && angleDiff <= sol4->maxAngleDiff) {
                double w = intendedMag * gCosineTableG[intendedDYaw >> 4];
                double eqB = (50.0 + 147200.0 / w);
                double eqC = -(320000.0 / w) * lower10KSpeed;
                double eqDet = eqB * eqB - eqC;
                float minSpeed = sqrt(eqDet) - eqB;

                eqC = -(320000.0 / w) * upper10KSpeed;
                eqDet = eqB * eqB - eqC;
                float maxSpeed = sqrt(eqDet) - eqB;

                int solIdx = atomicAdd(&nSK6Solutions, 1);

                if (solIdx < MAX_SK_PHASE_SIX) {
                    struct SKPhase6* solution = &(sk6Solutions[solIdx]);
                    solution->p5Idx = idx;
                    solution->minPre10KSpeed = minSpeed;
                    solution->maxPre10KSpeed = maxSpeed;
                    solution->minPost10KSpeed = lower10KSpeed;
                    solution->maxPost10KSpeed = upper10KSpeed;
                    solution->angleDiff = angleDiff;
                }
            }
        }
    }
}

__global__ void try_slide_kick_routeG2() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nSK4Solutions, MAX_SK_PHASE_FOUR)) {
        struct SKPhase4* sol4 = &(sk4Solutions[idx]);
        struct SKPhase3* sol3 = &(sk3Solutions[sol4->p3Idx]);
        struct SKPhase2* sol2 = (sol3->p2Type / 2 == 0) ? ((sol3->p2Type % 2 == 0) ? &(sk2ASolutions[sol3->p2Idx]) : &(sk2BSolutions[sol3->p2Idx])) : ((sol3->p2Type % 2 == 0) ? &(sk2CSolutions[sol3->p2Idx]) : &(sk2DSolutions[sol3->p2Idx]));
        struct SKPhase1* sol1 = &(sk1Solutions[sol2->p1Idx]);

        double minStickX = INFINITY;
        double maxStickX = -INFINITY;
        double minStickY = INFINITY;
        double maxStickY = -INFINITY;

        double stickBounds[4][2];

        for (int j = sol1->minF1AngleIdx; j <= sol1->maxF1AngleIdx; j++) {
            int f1Angle = (65536 + gArctanTableG[j % 8192]) % 65536;

            for (int i = 0; i < 4; i++) {
                double m1;
                double n1;

                if (i % 2 == 0) {
                    m1 = sol4->minM1;
                }
                else {
                    m1 = sol4->maxM1;
                }

                if (i / 2 == 0) {
                    n1 = sol4->minN1;
                }
                else {
                    n1 = sol4->maxN1;
                }

                double targetDYaw = 65536.0 * (atan2(n1, m1) / (2.0 * M_PI));
                double targetMag = sqrtf(m1 * m1 + n1 * n1);

                double stickAngle = fmod(65536.0 + fmod(targetDYaw + f1Angle - sol4->cameraYaw, 65536.0), 65536.0);
                double stickMagnitude = sqrt(128.0 * targetMag);

                double xS;
                double yS;

                if (stickMagnitude < 64.0) {
                    yS = -stickMagnitude * cos(2.0 * M_PI * (stickAngle / 65536.0));
                    xS = stickMagnitude * sin(2.0 * M_PI * (stickAngle / 65536.0));

                    minStickX = fmin(minStickX, xS);
                    minStickY = fmin(minStickY, yS);
                    maxStickX = fmax(maxStickX, xS);
                    maxStickY = fmax(maxStickY, yS);
                }
                else {
                    if (stickAngle <= 8192.0 || stickAngle > 57344.0) {
                        yS = -122.0;
                        xS = -122.0 * tan(2.0 * M_PI * (stickAngle / 65536.0));

                        minStickX = fmin(minStickX, xS);
                        minStickY = fmin(minStickY, yS);

                        yS = 64.0 * cos(2.0 * M_PI * (stickAngle / 65536.0));
                        xS = 64.0 * sin(2.0 * M_PI * (stickAngle / 65536.0));

                        maxStickX = fmax(maxStickX, xS);
                        maxStickY = fmax(maxStickY, yS);
                    }
                    else if (stickAngle > 8192.0 && stickAngle <= 24576.0) {
                        yS = 64.0 * cos(2.0 * M_PI * (stickAngle / 65536.0));
                        xS = 64.0 * sin(2.0 * M_PI * (stickAngle / 65536.0));

                        minStickX = fmin(minStickX, xS);
                        minStickY = fmin(minStickY, yS);

                        yS = 121.0 / tan(2.0 * M_PI * (stickAngle / 65536.0));
                        xS = 121.0;

                        maxStickX = fmax(maxStickX, xS);
                        maxStickY = fmax(maxStickY, yS);
                    }
                    else if (stickAngle > 24576.0 && stickAngle <= 40960.0) {
                        yS = 64.0 * cos(2.0 * M_PI * (stickAngle / 65536.0));
                        xS = 64.0 * sin(2.0 * M_PI * (stickAngle / 65536.0));

                        minStickX = fmin(minStickX, xS);
                        minStickY = fmin(minStickY, yS);

                        yS = 121.0;
                        xS = 121.0 * tan(2.0 * M_PI * (stickAngle / 65536.0));

                        maxStickX = fmax(maxStickX, xS);
                        maxStickY = fmax(maxStickY, yS);
                    }
                    else {
                        yS = -122.0 / tan(2.0 * M_PI * (stickAngle / 65536.0));
                        xS = -122.0;

                        minStickX = fmin(minStickX, xS);
                        minStickY = fmin(minStickY, yS);

                        yS = 64.0 * cos(2.0 * M_PI * (stickAngle / 65536.0));
                        xS = 64.0 * sin(2.0 * M_PI * (stickAngle / 65536.0));

                        maxStickX = fmax(maxStickX, xS);
                        maxStickY = fmax(maxStickY, yS);
                    }
                }

                if (maxStickX - minStickX < maxStickY - minStickY) {
                    for (int x = (int)ceil(minStickX); x <= (int)floor(maxStickX); x++) {
                        if (x != 1) {
                            int y = (int)round(((double)x - minStickX) * (maxStickY - minStickY) / (maxStickX - minStickX) + minStickY);

                            if (y != 1) {
                                int solIdx = atomicAdd(&nSK5Solutions, 1);

                                if (solIdx < MAX_SK_PHASE_FIVE) {
                                    struct SKPhase5* solution = &(sk5Solutions[solIdx]);
                                    solution->p4Idx = idx;
                                    solution->stickX = x;
                                    solution->stickY = y;
                                    solution->f1Angle = f1Angle;
                                }
                            }
                        }
                    }
                }
                else {
                    for (int y = (int)ceil(minStickY); y <= (int)floor(maxStickY); y++) {
                        if (y != 1) {
                            int x = (int)round(((double)y - minStickY) * (maxStickX - minStickX) / (maxStickY - minStickY) + minStickX);

                            if (x != 1) {
                                int solIdx = atomicAdd(&nSK5Solutions, 1);

                                if (solIdx < MAX_SK_PHASE_FIVE) {
                                    struct SKPhase5* solution = &(sk5Solutions[solIdx]);
                                    solution->p4Idx = idx;
                                    solution->stickX = x;
                                    solution->stickY = y;
                                    solution->f1Angle = f1Angle;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

__global__ void try_slide_kick_routeG(short* pyramidFloorPoints, const int nPoints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nSK3Solutions, MAX_SK_PHASE_THREE)) {
        struct SKPhase3* sol3 = &(sk3Solutions[idx]);
        struct SKPhase2* sol2 = (sol3->p2Type / 2 == 0) ? ((sol3->p2Type % 2 == 0) ? &(sk2ASolutions[sol3->p2Idx]) : &(sk2BSolutions[sol3->p2Idx])) : ((sol3->p2Type % 2 == 0) ? &(sk2CSolutions[sol3->p2Idx]) : &(sk2DSolutions[sol3->p2Idx]));
        struct SKPhase1* sol1 = &(sk1Solutions[sol2->p1Idx]);

        float cameraPositions[4][3] = { {-8192, -2918, -8192}, {-8192, -2918, 8191}, {8191, -2918, -8192}, {8191, -2918, 8191} };

        float tenKPosition[3] = { (65536.0 * sol3->x2) + (tenKFloors[sol2->tenKFloorIdx][0] + tenKFloors[sol2->tenKFloorIdx][1]) / 2.0f, (tenKFloors[sol2->tenKFloorIdx][4] + tenKFloors[sol2->tenKFloorIdx][5]) / 2.0f, (65536.0 * sol3->z2) + (tenKFloors[sol2->tenKFloorIdx][2] + tenKFloors[sol2->tenKFloorIdx][3]) / 2.0f };

        double minF2Dist = INFINITY;
        double maxF2Dist = -INFINITY;

        for (int i = 0; i < nPoints; i++) {
            for (int j = 0; j < 4; j++) {
                double PX = 65536.0 * sol1->x1 + pyramidFloorPoints[3 * i];
                double PZ = 65536.0 * sol1->z1 + pyramidFloorPoints[3 * i + 2];
                double QX1 = 65536.0 * sol3->x2 + ((j / 2 == 0) ? tenKFloors[sol2->tenKFloorIdx][0] : tenKFloors[sol2->tenKFloorIdx][1]);
                double QX2 = 65536.0 * sol3->x2 + ((((j + 1) % 4) / 2 == 0) ? tenKFloors[sol2->tenKFloorIdx][0] : tenKFloors[sol2->tenKFloorIdx][1]);
                double QZ1 = 65536.0 * sol3->z2 + ((((j + 1) % 4) / 2 == 0) ? tenKFloors[sol2->tenKFloorIdx][3] : tenKFloors[sol2->tenKFloorIdx][2]);
                double QZ2 = 65536.0 * sol3->z2 + ((j / 2 == 0) ? tenKFloors[sol2->tenKFloorIdx][3] : tenKFloors[sol2->tenKFloorIdx][2]);

                double s = ((QZ1 - PZ) * gSineTableG[sol2->f2Angle >> 4] - (QX1 - PX) * gCosineTableG[sol2->f2Angle >> 4]) / ((QX2 - QX1) * gCosineTableG[sol2->f2Angle >> 4] - (QZ2 - QZ1) * gSineTableG[sol2->f2Angle >> 4]);

                if (s >= 0.0 && s <= 1.0) {
                    double dist;

                    if (fabs(gSineTableG[sol2->f2Angle >> 4]) > fabs(gCosineTableG[sol2->f2Angle >> 4])) {
                        dist = (s * (QX2 - QX1) - (PX - QX1)) / gSineTableG[sol2->f2Angle >> 4];
                    }
                    else {
                        dist = (s * (QZ2 - QZ1) - (PZ - QZ1)) / gCosineTableG[sol2->f2Angle >> 4];
                    }

                    minF2Dist = fmin(minF2Dist, dist);
                    maxF2Dist = fmax(maxF2Dist, dist);
                }
            }
        }

        for (int i = 0; i < nPoints; i++) {
            for (int j = 0; j < 4; j++) {
                double PX = 65536.0 * sol3->x2 + ((j / 2 == 0) ? tenKFloors[sol2->tenKFloorIdx][0] : tenKFloors[sol2->tenKFloorIdx][1]);
                double PZ = 65536.0 * sol3->z2 + ((((j + 1) % 4) / 2 == 0) ? tenKFloors[sol2->tenKFloorIdx][0] : tenKFloors[sol2->tenKFloorIdx][1]);
                double QX1 = 65536.0 * sol1->x1 + pyramidFloorPoints[3 * i];
                double QX2 = 65536.0 * sol1->x1 + pyramidFloorPoints[3 * ((i + 1) % nPoints)];
                double QZ1 = 65536.0 * sol1->z1 + pyramidFloorPoints[3 * i + 2];
                double QZ2 = 65536.0 * sol1->z1 + pyramidFloorPoints[3 * ((i + 1) % nPoints) + 2];

                double s = ((QZ1 - PZ) * gSineTableG[sol2->f2Angle >> 4] - (QX1 - PX) * gCosineTableG[sol2->f2Angle >> 4]) / ((QX2 - QX1) * gCosineTableG[sol2->f2Angle >> 4] - (QZ2 - QZ1) * gSineTableG[sol2->f2Angle >> 4]);

                if (s >= 0.0 && s <= 1.0) {
                    double dist;

                    if (fabs(gSineTableG[sol2->f2Angle >> 4]) > fabs(gCosineTableG[sol2->f2Angle >> 4])) {
                        dist = -(s * (QX2 - QX1) - (PX - QX1)) / gSineTableG[sol2->f2Angle >> 4];
                    }
                    else {
                        dist = -(s * (QZ2 - QZ1) - (PZ - QZ1)) / gCosineTableG[sol2->f2Angle >> 4];
                    }

                    minF2Dist = fmin(minF2Dist, dist);
                    maxF2Dist = fmax(maxF2Dist, dist);
                }
            }
        }

        double minSpeed = fmaxf(sol1->minSpeed, 4.0 * minF2Dist / (float)sol1->q2);
        double maxSpeed = fminf(sol1->maxSpeed, 4.0 * maxF2Dist / (float)sol1->q2);

        if (minSpeed <= maxSpeed) {
            double minF3Dist = INFINITY;
            double maxF3Dist = -INFINITY;

            double minAngleDiff = INFINITY;
            double maxAngleDiff = -INFINITY;

            for (int i = 0; i < nPoints; i++) {
                for (int j = 0; j < 4; j++) {
                    double xDist;
                    double zDist;

                    if (j % 2 == 0) {
                        xDist = (65536.0 * sol3->x2 + tenKFloors[sol2->tenKFloorIdx][0]) - pyramidFloorPoints[3 * i];
                    }
                    else {
                        xDist = (65536.0 * sol3->x2 + tenKFloors[sol2->tenKFloorIdx][1]) - pyramidFloorPoints[3 * i];
                    }

                    if (j / 2 == 0) {
                        zDist = (65536.0 * sol3->z2 + tenKFloors[sol2->tenKFloorIdx][2]) - pyramidFloorPoints[3 * i + 2];
                    }
                    else {
                        zDist = (65536.0 * sol3->z2 + tenKFloors[sol2->tenKFloorIdx][3]) - pyramidFloorPoints[3 * i + 2];
                    }

                    double dist = sqrt(xDist * xDist + zDist * zDist);

                    minF3Dist = fmin(minF3Dist, dist);
                    maxF3Dist = fmax(maxF3Dist, dist);

                    double f3Angle = 65536.0 * atan2(xDist, zDist) / (2.0 * M_PI);

                    double angleDiff = fmod(65536.0 + f3Angle - sol2->f2Angle, 65536.0);
                    angleDiff = fmod(angleDiff + 32768.0, 65536.0) - 32768.0;

                    minAngleDiff = fmin(minAngleDiff, angleDiff);
                    maxAngleDiff = fmax(maxAngleDiff, angleDiff);
                }
            }

            minAngleDiff = fmax(minAngleDiff, -(double)maxF3Turn);
            maxAngleDiff = fmin(maxAngleDiff, (double)maxF3Turn);

            if (minAngleDiff <= maxAngleDiff) {
                double minF3Angle = minAngleDiff + sol2->f2Angle;
                double maxF3Angle = maxAngleDiff + sol2->f2Angle;

                double minN;
                double maxN;

                if (sol2->f2Angle == 0 || sol2->f2Angle == 32768) {
                    double sinF2Angle = sin(2.0 * M_PI * (double)sol2->f2Angle / 65536.0);

                    minN = -cos(2.0 * M_PI * minF3Angle / 65536.0) / sinF2Angle;
                    maxN = -cos(2.0 * M_PI * maxF3Angle / 65536.0) / sinF2Angle;
                }
                else {
                    double sinF2Angle = gSineTableG[sol2->f2Angle >> 4];
                    double cosF2Angle = gCosineTableG[sol2->f2Angle >> 4];

                    double sinMinF3Angle = sin(2.0 * M_PI * minF3Angle / 65536.0);
                    double cosMinF3Angle = cos(2.0 * M_PI * minF3Angle / 65536.0);

                    double sinMaxF3Angle = sin(2.0 * M_PI * maxF3Angle / 65536.0);
                    double cosMaxF3Angle = cos(2.0 * M_PI * maxF3Angle / 65536.0);

                    double t = sinF2Angle / cosF2Angle;
                    double s = sinMinF3Angle / cosMinF3Angle;

                    bool signTest = (cosF2Angle > 0 && cosMinF3Angle > 0) || (cosF2Angle < 0 && cosMinF3Angle < 0);

                    if (signTest) {
                        minN = (-((double)s * (double)t) - 1.0 + sqrt(((double)s * (double)t - 1.0) * ((double)s * (double)t - 1.0) + 4.0 * (double)s * (double)s)) / (2.0 * (double)s);
                    }
                    else {
                        minN = (-((double)s * (double)t) - 1.0 - sqrt(((double)s * (double)t - 1.0) * ((double)s * (double)t - 1.0) + 4.0 * (double)s * (double)s)) / (2.0 * (double)s);
                    }

                    s = sinMaxF3Angle / cosMaxF3Angle;

                    signTest = (cosF2Angle > 0 && cosMaxF3Angle > 0) || (cosF2Angle < 0 && cosMaxF3Angle < 0);

                    if (signTest) {
                        maxN = (-((double)s * (double)t) - 1.0 + sqrt(((double)s * (double)t - 1.0) * ((double)s * (double)t - 1.0) + 4.0 * (double)s * (double)s)) / (2.0 * (double)s);
                    }
                    else {
                        maxN = (-((double)s * (double)t) - 1.0 - sqrt(((double)s * (double)t - 1.0) * ((double)s * (double)t - 1.0) + 4.0 * (double)s * (double)s)) / (2.0 * (double)s);
                    }
                }

                double minN1 = 32.0 * minN / 0.05;
                double maxN1 = 32.0 * maxN / 0.05;

                if (minN1 > maxN1) {
                    double temp = minN1;
                    minN1 = maxN1;
                    maxN1 = temp;
                }

                minN1 = fmax(minN1, -32.0);
                maxN1 = fmin(maxN1, 32.0);

                if (minN1 <= maxN1) {
                    float minPost10KSpeed = -4.0 * minF3Dist / tenKFloors[sol2->tenKFloorIdx][7];
                    float maxPost10KSpeed = -4.0 * maxF3Dist / tenKFloors[sol2->tenKFloorIdx][7];

                    double minM = (double)minPost10KSpeed / (double)maxSpeed;
                    double maxM = (double)maxPost10KSpeed / (double)minSpeed;

                    double minM1 = 32.0 * ((minM - 0.92) / 0.02) / (double)(0.5f + (0.5f * maxSpeed / 100.0f));
                    double maxM1 = 32.0 * ((maxM - 0.92) / 0.02) / (double)(0.5f + (0.5f * minSpeed / 100.0f));

                    if (minM1 > maxM1) {
                        double temp = minM1;
                        minM1 = maxM1;
                        maxM1 = temp;
                    }

                    minM1 = fmax(minM1, -32.0);
                    maxM1 = fmin(maxM1, 0.0);

                    if (minM1 <= maxM1) {
                        int minCameraYaw = 0;
                        int maxCameraYaw = 0;

                        float cameraFocus[3] = { 0.0f, 0.0f, 0.0f };

                        for (int i = 0; i < nPoints; i++) {
                            cameraFocus[0] += pyramidFloorPoints[3 * i];
                            cameraFocus[1] += pyramidFloorPoints[3 * i + 1];
                            cameraFocus[2] += pyramidFloorPoints[3 * i + 2];
                        }

                        cameraFocus[0] /= nPoints;
                        cameraFocus[1] /= nPoints;
                        cameraFocus[2] /= nPoints;

                        cameraFocus[0] += 0.8 * 65536.0 * sol1->x1;
                        cameraFocus[2] += 0.8 * 65536.0 * sol1->z1;

                        int refCameraYaw = calculate_camera_yaw(cameraFocus, cameraPositions[0]);
                        refCameraYaw = (65536 + refCameraYaw) % 65536;

                        for (int k = 1; k < 4; k++) {
                            int cameraYaw = calculate_camera_yaw(cameraFocus, cameraPositions[k]);
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
                                int solIdx = atomicAdd(&nSK4Solutions, 1);

                                if (solIdx < MAX_SK_PHASE_FOUR) {
                                    struct SKPhase4* solution = &(sk4Solutions[solIdx]);
                                    solution->p3Idx = idx;
                                    solution->cameraYaw = cameraYaw;
                                    solution->minM1 = minM1;
                                    solution->maxM1 = maxM1;
                                    solution->minN1 = minN1;
                                    solution->maxN1 = maxN1;
                                    solution->minPre10KSpeed = minSpeed;
                                    solution->maxPre10KSpeed = maxSpeed;
                                    solution->minPost10KSpeed = minPost10KSpeed;
                                    solution->maxPost10KSpeed = maxPost10KSpeed;
                                    solution->minAngleDiff = minAngleDiff;
                                    solution->maxAngleDiff = maxAngleDiff;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

__global__ void find_slide_kick_setupG3a(float platformMinZ, float platformMaxZ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nSK2ASolutions, MAX_SK_PHASE_TWO_A)) {
        struct SKPhase2* sol2 = &(sk2CSolutions[idx]);
        struct SKPhase1* sol1 = &(sk1Solutions[sol2->p1Idx]);

        int cosSign = (sol2->cosAngle > 0) - (sol2->cosAngle < 0);

        double speed1 = ((cosSign + 1) >> 1) * sol1->minSpeed + (((cosSign + 1) >> 1) ^ 1) * sol1->maxSpeed;
        double speed2 = ((cosSign + 1) >> 1) * sol1->maxSpeed + (((cosSign + 1) >> 1) ^ 1) * sol1->minSpeed;

        int minF2ZPU = (int)ceil((65536.0 * sol1->z1 + platformMinZ + speed1 * (double)sol1->q2 / 4.0 - tenKFloors[sol2->tenKFloorIdx][3]) / 65536.0);
        int maxF2ZPU = (int)floor((65536.0 * sol1->z1 + platformMaxZ + speed2 * (double)sol1->q2 / 4.0 - tenKFloors[sol2->tenKFloorIdx][2]) / 65536.0);

        minF2ZPU += (sol1->q2 + ((sol1->z1 - minF2ZPU) % sol1->q2)) % sol1->q2;
        maxF2ZPU -= (sol1->q2 + ((minF2ZPU - sol1->z1) % sol1->q2)) % sol1->q2;

        for (int z2 = minF2ZPU; z2 <= maxF2ZPU; z2 += sol1->q2) {
            int solIdx = atomicAdd(&nSK3Solutions, 1);

            if (solIdx < MAX_SK_PHASE_THREE) {
                struct SKPhase3* solution = &(sk3Solutions[solIdx]);
                solution->p2Idx = idx;
                solution->p2Type = 0;
                solution->x2 = sol1->x1;
                solution->z2 = z2;
            }
        }
    }
}

__global__ void find_slide_kick_setupG3b(float platformMinX, float platformMaxX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nSK2BSolutions, MAX_SK_PHASE_TWO_B)) {
        struct SKPhase2* sol2 = &(sk2CSolutions[idx]);
        struct SKPhase1* sol1 = &(sk1Solutions[sol2->p1Idx]);

        int sinSign = (sol2->sinAngle > 0) - (sol2->sinAngle < 0);

        double speed1 = ((sinSign + 1) >> 1) * sol1->minSpeed + (((sinSign + 1) >> 1) ^ 1) * sol1->maxSpeed;
        double speed2 = ((sinSign + 1) >> 1) * sol1->maxSpeed + (((sinSign + 1) >> 1) ^ 1) * sol1->minSpeed;

        int minF2XPU = (int)ceil((65536.0 * sol1->x1 + platformMinX + speed1 * (double)sol1->q2 / 4.0 - tenKFloors[sol2->tenKFloorIdx][1]) / 65536.0);
        int maxF2XPU = (int)floor((65536.0 * sol1->x1 + platformMaxX + speed2 * (double)sol1->q2 / 4.0 - tenKFloors[sol2->tenKFloorIdx][0]) / 65536.0);

        minF2XPU += (sol1->q2 + ((sol1->x1 - minF2XPU) % sol1->q2)) % sol1->q2;
        maxF2XPU -= (sol1->q2 + ((minF2XPU - sol1->x1) % sol1->q2)) % sol1->q2;

        for (int x2 = minF2XPU; x2 <= maxF2XPU; x2 += sol1->q2) {
            int solIdx = atomicAdd(&nSK3Solutions, 1);

            if (solIdx < MAX_SK_PHASE_THREE) {
                struct SKPhase3* solution = &(sk3Solutions[solIdx]);
                solution->p2Idx = idx;
                solution->p2Type = 1;
                solution->x2 = x2;
                solution->z2 = sol1->z1;
            }
        }
    }
}

__global__ void find_slide_kick_setupG3c(float platformMinX, float platformMaxX, float platformMinZ, float platformMaxZ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nSK2CSolutions, MAX_SK_PHASE_TWO_C)) {
        struct SKPhase2* sol2 = &(sk2CSolutions[idx]);
        struct SKPhase1* sol1 = &(sk1Solutions[sol2->p1Idx]);

        double cotAngle = sol2->cosAngle / sol2->sinAngle;

        int sinSign = (sol2->sinAngle > 0) - (sol2->sinAngle < 0);
        int cosSign = (sol2->cosAngle > 0) - (sol2->cosAngle < 0);
        int cotSign = (cotAngle > 0) - (cotAngle < 0);

        double speed1 = ((sinSign + 1) >> 1) * sol1->minSpeed + (((sinSign + 1) >> 1) ^ 1) * sol1->maxSpeed;
        double speed2 = ((sinSign + 1) >> 1) * sol1->maxSpeed + (((sinSign + 1) >> 1) ^ 1) * sol1->minSpeed;

        int minF2XPU = (int)ceil((65536.0 * sol1->x1 + platformMinX + speed1 * sol2->sinAngle * (double)sol1->q2 / 4.0 - tenKFloors[sol2->tenKFloorIdx][1]) / 65536.0);
        int maxF2XPU = (int)floor((65536.0 * sol1->x1 + platformMaxX + speed2 * sol2->sinAngle * (double)sol1->q2 / 4.0 - tenKFloors[sol2->tenKFloorIdx][0]) / 65536.0);

        minF2XPU += (sol1->q2 + ((sol1->x1 - minF2XPU) % sol1->q2)) % sol1->q2;
        maxF2XPU -= (sol1->q2 + ((minF2XPU - sol1->x1) % sol1->q2)) % sol1->q2;

        speed1 = ((cosSign + 1) >> 1) * sol1->minSpeed + (((cosSign + 1) >> 1) ^ 1) * sol1->maxSpeed;
        speed2 = ((cosSign + 1) >> 1) * sol1->maxSpeed + (((cosSign + 1) >> 1) ^ 1) * sol1->minSpeed;

        int minF2ZPU = (int)ceil((65536.0 * sol1->z1 + platformMinZ + speed1 * sol2->cosAngle * (double)sol1->q2 / 4.0 - tenKFloors[sol2->tenKFloorIdx][3]) / 65536.0);
        int maxF2ZPU = (int)floor((65536.0 * sol1->z1 + platformMaxZ + speed2 * sol2->cosAngle * (double)sol1->q2 / 4.0 - tenKFloors[sol2->tenKFloorIdx][2]) / 65536.0);

        minF2ZPU += (sol1->q2 + ((sol1->z1 - minF2ZPU) % sol1->q2)) % sol1->q2;
        maxF2ZPU -= (sol1->q2 + ((minF2ZPU - sol1->z1) % sol1->q2)) % sol1->q2;

        int floorPointIdx = 1 - ((cotSign + 1) >> 1);
        float tenKFloorX1 = tenKFloors[sol2->tenKFloorIdx][floorPointIdx];
        float tenKFloorX2 = tenKFloors[sol2->tenKFloorIdx][1 - floorPointIdx];
        float platformX1 = ((cotSign + 1) >> 1) * platformMaxX + (((cotSign + 1) >> 1) ^ 1) * platformMinX;
        float platformX2 = ((cotSign + 1) >> 1) * platformMinX + (((cotSign + 1) >> 1) ^ 1) * platformMaxX;
        float zRange1 = ((cotSign + 1) >> 1) * sol2->lower + (((cotSign + 1) >> 1) ^ 1) * sol2->upper;
        float zRange2 = ((cotSign + 1) >> 1) * sol2->upper + (((cotSign + 1) >> 1) ^ 1) * sol2->lower;

        for (int x2 = minF2XPU; x2 <= maxF2XPU; x2 += sol1->q2) {
            int minF2XZPU = (int)ceil((65536.0 * sol1->z1 + platformMinZ + ((65536.0 * x2 + tenKFloorX1) - (65536.0 * sol1->x1 + platformX1)) * cotAngle + zRange1 - tenKFloors[sol2->tenKFloorIdx][3]) / 65536.0);
            int maxF2XZPU = (int)floor((65536.0 * sol1->z1 + platformMaxZ + ((65536.0 * x2 + tenKFloorX2) - (65536.0 * sol1->x1 + platformX2)) * cotAngle + zRange2 - tenKFloors[sol2->tenKFloorIdx][2]) / 65536.0);

            minF2XZPU += (sol1->q2 + ((sol1->z1 - minF2XZPU) % sol1->q2)) % sol1->q2;
            maxF2XZPU -= (sol1->q2 + ((maxF2XZPU - sol1->z1) % sol1->q2)) % sol1->q2;

            minF2XZPU = max(minF2XZPU, minF2ZPU);
            maxF2XZPU = min(maxF2XZPU, maxF2ZPU);

            for (int z2 = minF2ZPU; z2 <= maxF2ZPU; z2 += sol1->q2) {
                int solIdx = atomicAdd(&nSK3Solutions, 1);

                if (solIdx < MAX_SK_PHASE_THREE) {
                    struct SKPhase3* solution = &(sk3Solutions[solIdx]);
                    solution->p2Idx = idx;
                    solution->p2Type = 2;
                    solution->x2 = x2;
                    solution->z2 = z2;
                }
            }
        }
    }
}

__global__ void find_slide_kick_setupG3d(float platformMinX, float platformMaxX, float platformMinZ, float platformMaxZ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nSK2DSolutions, MAX_SK_PHASE_TWO_D)) {
        struct SKPhase2* sol2 = &(sk2DSolutions[idx]);
        struct SKPhase1* sol1 = &(sk1Solutions[sol2->p1Idx]);

        double tanAngle = sol2->sinAngle / sol2->cosAngle;

        int sinSign = (sol2->sinAngle > 0) - (sol2->sinAngle < 0);
        int cosSign = (sol2->cosAngle > 0) - (sol2->cosAngle < 0);
        int tanSign = (tanAngle > 0) - (tanAngle < 0);

        double speed1 = ((sinSign + 1) >> 1) * sol1->minSpeed + (((sinSign + 1) >> 1) ^ 1) * sol1->maxSpeed;
        double speed2 = ((sinSign + 1) >> 1) * sol1->maxSpeed + (((sinSign + 1) >> 1) ^ 1) * sol1->minSpeed;

        int minF2XPU = (int)ceil((65536.0 * sol1->x1 + platformMinX + speed1 * sol2->sinAngle * (double)sol1->q2 / 4.0 - tenKFloors[sol2->tenKFloorIdx][1]) / 65536.0);
        int maxF2XPU = (int)floor((65536.0 * sol1->x1 + platformMaxX + speed2 * sol2->sinAngle * (double)sol1->q2 / 4.0 - tenKFloors[sol2->tenKFloorIdx][0]) / 65536.0);

        minF2XPU += (sol1->q2 + ((sol1->x1 - minF2XPU) % sol1->q2)) % sol1->q2;
        maxF2XPU -= (sol1->q2 + ((minF2XPU - sol1->x1) % sol1->q2)) % sol1->q2;

        speed1 = ((cosSign + 1) >> 1) * sol1->minSpeed + (((cosSign + 1) >> 1) ^ 1) * sol1->maxSpeed;
        speed2 = ((cosSign + 1) >> 1) * sol1->maxSpeed + (((cosSign + 1) >> 1) ^ 1) * sol1->minSpeed;

        int minF2ZPU = (int)ceil((65536.0 * sol1->z1 + platformMinZ + speed1 * sol2->cosAngle * (double)sol1->q2 / 4.0 - tenKFloors[sol2->tenKFloorIdx][3]) / 65536.0);
        int maxF2ZPU = (int)floor((65536.0 * sol1->z1 + platformMaxZ + speed2 * sol2->cosAngle * (double)sol1->q2 / 4.0 - tenKFloors[sol2->tenKFloorIdx][2]) / 65536.0);

        minF2ZPU += (sol1->q2 + ((sol1->z1 - minF2ZPU) % sol1->q2)) % sol1->q2;
        maxF2ZPU -= (sol1->q2 + ((minF2ZPU - sol1->z1) % sol1->q2)) % sol1->q2;

        int floorPointIdx = 3 - ((tanSign + 1) >> 1);
        float tenKFloorZ1 = tenKFloors[sol2->tenKFloorIdx][floorPointIdx];
        float tenKFloorZ2 = tenKFloors[sol2->tenKFloorIdx][5 - floorPointIdx];
        float platformZ1 = ((tanSign + 1) >> 1) * platformMaxZ + (((tanSign + 1) >> 1) ^ 1) * platformMinZ;
        float platformZ2 = ((tanSign + 1) >> 1) * platformMinZ + (((tanSign + 1) >> 1) ^ 1) * platformMaxZ;
        float xRange1 = ((tanSign + 1) >> 1) * sol2->lower + (((tanSign + 1) >> 1) ^ 1) * sol2->upper;
        float xRange2 = ((tanSign + 1) >> 1) * sol2->upper + (((tanSign + 1) >> 1) ^ 1) * sol2->lower;

        for (int z2 = minF2ZPU; z2 <= maxF2ZPU; z2 += sol1->q2) {
            int minF2ZXPU = (int)ceil((65536.0 * sol1->x1 + platformMinX + ((65536.0 * z2 + tenKFloorZ1) - (65536.0 * sol1->z1 + platformZ1)) * tanAngle + xRange1 - tenKFloors[sol2->tenKFloorIdx][1]) / 65536.0);
            int maxF2ZXPU = (int)floor((65536.0 * sol1->x1 + platformMaxX + ((65536.0 * z2 + tenKFloorZ2) - (65536.0 * sol1->z1 + platformZ2)) * tanAngle + xRange2 - tenKFloors[sol2->tenKFloorIdx][0]) / 65536.0);

            minF2ZXPU += (sol1->q2 + ((sol1->x1 - minF2ZXPU) % sol1->q2)) % sol1->q2;
            maxF2ZXPU -= (sol1->q2 + ((maxF2ZXPU - sol1->x1) % sol1->q2)) % sol1->q2;

            minF2ZXPU = max(minF2ZXPU, minF2XPU);
            maxF2ZXPU = min(maxF2ZXPU, maxF2XPU);

            for (int x2 = minF2ZXPU; x2 <= maxF2ZXPU; x2 += sol1->q2) {
                int solIdx = atomicAdd(&nSK3Solutions, 1);

                if (solIdx < MAX_SK_PHASE_THREE) {
                    struct SKPhase3* solution = &(sk3Solutions[solIdx]);
                    solution->p2Idx = idx;
                    solution->p2Type = 3;
                    solution->x2 = x2;
                    solution->z2 = z2;
                }
            }
        }
    }
}

__global__ void find_slide_kick_setupG2(short* floorPoints, const int nPoints, float floorNormalY, float platformMinX, float platformMaxX, float platformMinZ, float platformMaxZ, float midPointX, float midPointZ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nSK1Solutions, MAX_SK_PHASE_ONE)) {
        struct SKPhase1* sol = &(sk1Solutions[idx]);
        double puAngle = 65536.0 * atan2((double)sol->x1, (double)sol->z1) / (2.0 * M_PI);
        puAngle = fmod(65536.0 + puAngle, 65536.0);

        int puAngleClosest = (65536 + atan2sG(sol->z1, sol->x1)) % 65536;

        double sinMaxAngle = sin(2.0 * M_PI * (double)maxF3Turn / 65536.0);
        double maxF2AngleChange = fmod(32768.0 - (65536.0 * asin(sol->q2 * sinMaxAngle / (4.0 * floorNormalY)) / (2.0 * M_PI)) - maxF3Turn, 32768.0);
        maxF2AngleChange = fabs(fmod(maxF2AngleChange + 16384.0, 32768.0) - 16384.0);

        int minF2AngleIdx = gReverseArctanTable[puAngleClosest];
        int maxF2AngleIdx = gReverseArctanTable[puAngleClosest];

        while ((65536 + puAngleClosest - ((gArctanTableG[(minF2AngleIdx + 8191) % 8192] >> 4) << 4)) % 65536 < maxF2AngleChange) {
            minF2AngleIdx = minF2AngleIdx - 1;
        }

        while ((65536 + ((gArctanTableG[(maxF2AngleIdx + 1) % 8192] >> 4) << 4) - puAngleClosest) % 65536 < maxF2AngleChange) {
            maxF2AngleIdx = maxF2AngleIdx + 1;
        }

        for (int a = minF2AngleIdx; a <= maxF2AngleIdx; a++) {
            int f2Angle = gArctanTableG[(8192 + a) % 8192];
            f2Angle = (65536 + f2Angle) % 65536;

            if (f2Angle == 0 || f2Angle == 32768) {
                for (int i = 0; i < nTenKFloors; i++) {
                    float minX = fmaxf(platformMinX, tenKFloors[i][0]);
                    float maxX = fminf(platformMaxX, tenKFloors[i][1]);

                    if (minX < maxX) {
                        int solIdx = atomicAdd(&nSK2ASolutions, 1);

                        if (solIdx < MAX_SK_PHASE_TWO_A) {
                            struct SKPhase2* solution = &(sk2ASolutions[solIdx]);
                            solution->p1Idx = idx;
                            solution->f2Angle = 0;
                            solution->tenKFloorIdx = i;
                            solution->sinAngle = 0.0;
                            solution->cosAngle = 1.0;
                        }

                        solIdx = atomicAdd(&nSK2ASolutions, 1);

                        if (solIdx < MAX_SK_PHASE_TWO_A) {
                            struct SKPhase2* solution = &(sk2ASolutions[solIdx]);
                            solution->p1Idx = idx;
                            solution->f2Angle = 32768;
                            solution->tenKFloorIdx = i;
                            solution->sinAngle = 0.0;
                            solution->cosAngle = -1.0;
                        }
                    }
                }
            }
            else if (f2Angle == 16384 || f2Angle == 49152) {
                for (int i = 0; i < nTenKFloors; i++) {
                    float minZ = fmaxf(platformMinZ, tenKFloors[i][2]);
                    float maxZ = fminf(platformMaxZ, tenKFloors[i][3]);

                    if (minZ < maxZ) {
                        int solIdx = atomicAdd(&nSK2BSolutions, 1);

                        if (solIdx < MAX_SK_PHASE_TWO_B) {
                            struct SKPhase2* solution = &(sk2BSolutions[solIdx]);
                            solution->p1Idx = idx;
                            solution->f2Angle = 16384;
                            solution->tenKFloorIdx = i;
                            solution->sinAngle = 1.0;
                            solution->cosAngle = 0.0;
                        }

                        solIdx = atomicAdd(&nSK2BSolutions, 1);

                        if (solIdx < MAX_SK_PHASE_TWO_B) {
                            struct SKPhase2* solution = &(sk2BSolutions[solIdx]);
                            solution->p1Idx = idx;
                            solution->f2Angle = 49152;
                            solution->tenKFloorIdx = i;
                            solution->sinAngle = 1.0;
                            solution->cosAngle = 0.0;
                        }
                    }
                }
            }
            else {
                double sinAngle = gSineTableG[f2Angle >> 4];
                double cosAngle = gCosineTableG[f2Angle >> 4];

                if (fabs(sinAngle) < fabs(cosAngle)) {
                    float lowerZ = INFINITY;
                    float upperZ = -INFINITY;

                    double cotAngle = cosAngle / sinAngle;

                    for (int i = 0; i < nPoints; i++) {
                        float testZ = floorPoints[3 * i + 2] + ((midPointX - floorPoints[3 * i]) * cotAngle) - midPointZ;
                        lowerZ = fminf(lowerZ, testZ);
                        upperZ = fmaxf(upperZ, testZ);
                    }

                    for (int i = 0; i < nTenKFloors; i++) {
                        int solIdx = atomicAdd(&nSK2CSolutions, 1);

                        if (solIdx < MAX_SK_PHASE_TWO_C) {
                            struct SKPhase2* solution = &(sk2CSolutions[solIdx]);
                            solution->p1Idx = idx;
                            solution->f2Angle = f2Angle;
                            solution->tenKFloorIdx = i;
                            solution->lower = lowerZ;
                            solution->upper = upperZ;
                            solution->sinAngle = sinAngle;
                            solution->cosAngle = cosAngle;
                        }

                        solIdx = atomicAdd(&nSK2CSolutions, 1);

                        if (solIdx < MAX_SK_PHASE_TWO_C) {
                            struct SKPhase2* solution = &(sk2CSolutions[solIdx]);
                            solution->p1Idx = idx;
                            solution->f2Angle = (f2Angle + 32768) % 65536;
                            solution->tenKFloorIdx = i;
                            solution->lower = lowerZ;
                            solution->upper = upperZ;
                            solution->sinAngle = -sinAngle;
                            solution->cosAngle = -cosAngle;
                        }
                    }
                }
                else {
                    float lowerX = INFINITY;
                    float upperX = -INFINITY;

                    double tanAngle = sinAngle / cosAngle;

                    for (int i = 0; i < nPoints; i++) {
                        float testX = floorPoints[3 * i] + ((midPointZ - floorPoints[3 * i + 2]) * tanAngle) - midPointX;
                        lowerX = fminf(lowerX, testX);
                        upperX = fmaxf(upperX, testX);
                    }

                    for (int i = 0; i < nTenKFloors; i++) {
                        int solIdx = atomicAdd(&nSK2DSolutions, 1);

                        if (solIdx < MAX_SK_PHASE_TWO_D) {
                            struct SKPhase2* solution = &(sk2DSolutions[solIdx]);
                            solution->p1Idx = idx;
                            solution->f2Angle = f2Angle;
                            solution->tenKFloorIdx = i;
                            solution->lower = lowerX;
                            solution->upper = upperX;
                            solution->sinAngle = sinAngle;
                            solution->cosAngle = cosAngle;
                        }

                        solIdx = atomicAdd(&nSK2DSolutions, 1);

                        if (solIdx < MAX_SK_PHASE_TWO_D) {
                            struct SKPhase2* solution = &(sk2DSolutions[solIdx]);
                            solution->p1Idx = idx;
                            solution->f2Angle = (f2Angle + 32768) % 65536;
                            solution->tenKFloorIdx = i;
                            solution->lower = lowerX;
                            solution->upper = upperX;
                            solution->sinAngle = -sinAngle;
                            solution->cosAngle = -cosAngle;
                        }
                    }
                }
            }
        }
    }
}

__global__ void find_slide_kick_setupG(short* floorPoints, const int nPoints, float floorNormalY, double maxSpeed, int maxF1PU, int t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int x1 = 4 * (idx % (2 * (maxF1PU / 4) + 1)) - maxF1PU;
    int z1 = 4 * (idx / (2 * (maxF1PU / 4) + 1)) - maxF1PU;

    if ((x1 != 0 || z1 != 0) && 65536.0 * sqrt((double)(x1 * x1 + z1 * z1)) <= floorNormalY * maxSpeed) {
        float dx = 65536 * x1;
        float dy = 500.0f;
        float dz = 65536 * z1;

        float d = sqrtf(dx * dx + dy * dy + dz * dz);

        d = 1.0 / d;
        dx *= d;
        dy *= d;
        dz *= d;

        float normal_change[3];
        normal_change[0] = (platformNormal[0] <= dx) ? ((dx - platformNormal[0] < 0.01f) ? dx - platformNormal[0] : 0.01f) : ((dx - platformNormal[0] > -0.01f) ? dx - platformNormal[0] : -0.01f);
        normal_change[1] = (platformNormal[1] <= dy) ? ((dy - platformNormal[1] < 0.01f) ? dy - platformNormal[1] : 0.01f) : ((dy - platformNormal[1] > -0.01f) ? dy - platformNormal[1] : -0.01f);
        normal_change[2] = (platformNormal[2] <= dz) ? ((dz - platformNormal[2] < 0.01f) ? dz - platformNormal[2] : 0.01f) : ((dz - platformNormal[2] > -0.01f) ? dz - platformNormal[2] : -0.01f);

        if (normal_change[0] == normal_offsets[t][0] && normal_change[1] == normal_offsets[t][1] && normal_change[2] == normal_offsets[t][2]) {
            double qStepMul = (nPoints == 4) ? 1.0 : (4.0 / 3.0);

            double maxF1Dist = -INFINITY;
            double minF1Dist = INFINITY;

            int refAngle = 0;

            int maxF1Angle = -65536;
            int minF1Angle = 65536;

            for (int i = 0; i < nPoints; i++) {
                for (int j = 0; j < nPoints; j++) {
                    double xDist = (65536.0 * x1 + (floorPoints[3 * i] - floorPoints[3 * j]) * qStepMul);
                    double zDist = (65536.0 * z1 + (floorPoints[3 * i + 2] - floorPoints[3 * j + 2]) * qStepMul);

                    double dist = sqrt(xDist * xDist + zDist * zDist);

                    minF1Dist = fmin(minF1Dist, dist);
                    maxF1Dist = fmax(maxF1Dist, dist);

                    int angle = atan2sG(zDist, xDist);

                    if (i == 0 && j == 0) {
                        refAngle = angle;
                    }

                    angle = angle - refAngle;

                    minF1Angle = min(minF1Angle, angle);
                    maxF1Angle = max(maxF1Angle, angle);

                }
            }

            double minSpeedF1 = minF1Dist / floorNormalY;
            double maxSpeedF1 = fmin(maxSpeed, maxF1Dist / floorNormalY);

            if (minSpeedF1 < maxSpeedF1) {
                minF1Angle = (65536 + minF1Angle + refAngle) % 65536;
                maxF1Angle = (65536 + maxF1Angle + refAngle) % 65536;

                int minF1AngleIdx = gReverseArctanTable[minF1Angle];
                int maxF1AngleIdx = gReverseArctanTable[maxF1Angle];

                if (maxF1AngleIdx < minF1AngleIdx) {
                    maxF1AngleIdx = maxF1AngleIdx + 8192;
                }

                for (int q2 = 1; q2 <= 4; q2++) {
                    int solIdx = atomicAdd(&nSK1Solutions, 1);

                    if (solIdx < MAX_SK_PHASE_ONE) {
                        struct SKPhase1* solution = &(sk1Solutions[solIdx]);
                        solution->x1 = x1;
                        solution->z1 = z1;
                        solution->q2 = q2;
                        solution->minSpeed = minSpeedF1;
                        solution->maxSpeed = maxSpeedF1;
                        solution->minF1Dist = minF1Dist;
                        solution->maxF1Dist = maxF1Dist;
                        solution->minF1AngleIdx = minF1AngleIdx;
                        solution->maxF1AngleIdx = maxF1AngleIdx;
                    }
                }
            }
        }
    }
}

void find_slide_kick_setup_triangle(short* floorPoints, short* devFloorPoints, int nPoints, float yNormal, int t, double maxSpeed, int nThreads) {
    int nSK1SolutionsCPU = 0;
    int nSK2ASolutionsCPU = 0;
    int nSK2BSolutionsCPU = 0;
    int nSK2CSolutionsCPU = 0;
    int nSK2DSolutionsCPU = 0;
    int nSK3SolutionsCPU = 0;
    int nSK4SolutionsCPU = 0;
    int nSK5SolutionsCPU = 0;
    int nSK6SolutionsCPU = 0;

    cudaMemcpyToSymbol(nSK1Solutions, &nSK1SolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(nSK2ASolutions, &nSK2ASolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(nSK2BSolutions, &nSK2BSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(nSK2CSolutions, &nSK2CSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(nSK2DSolutions, &nSK2DSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(nSK3Solutions, &nSK3SolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(nSK4Solutions, &nSK4SolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(nSK5Solutions, &nSK5SolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(nSK6Solutions, &nSK6SolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

    float platformMinX = 32767.0;
    float platformMaxX = -32768.0;
    float platformMinZ = 32767.0;
    float platformMaxZ = -32768.0;

    for (int i = 0; i < nPoints; i++) {
        platformMinX = fminf(platformMinX, (float)floorPoints[3 * i]);
        platformMaxX = fmaxf(platformMaxX, (float)floorPoints[3 * i]);
    }

    for (int i = 0; i < nPoints; i++) {
        platformMinZ = fminf(platformMinZ, (float)floorPoints[3 * i + 2]);
        platformMaxZ = fmaxf(platformMaxZ, (float)floorPoints[3 * i + 2]);
    }

    float midPointX = 0.0f;
    float midPointZ = 0.0f;

    for (int i = 0; i < nPoints; i++) {
        midPointX += floorPoints[3 * i];
        midPointZ += floorPoints[3 * i + 2];
    }

    midPointX /= (float)nPoints;
    midPointZ /= (float)nPoints;

    cudaMemcpy(devFloorPoints, floorPoints, 3 * nPoints * sizeof(short), cudaMemcpyHostToDevice);

    int maxF1PU = (int)floor(yNormal * maxSpeed / (4.0 * 65536.0)) * 4;
    int nBlocks = ((2 * (maxF1PU / 4) + 1) * (2 * (maxF1PU / 4) + 1) + nThreads - 1) / nThreads;

    find_slide_kick_setupG << <nBlocks, nThreads >> > (devFloorPoints, nPoints, yNormal, maxSpeed, maxF1PU, t);

    cudaMemcpyFromSymbol(&nSK1SolutionsCPU, nSK1Solutions, sizeof(int), 0, cudaMemcpyDeviceToHost);

    if (nSK1SolutionsCPU > 0) {
        if (nSK1SolutionsCPU > MAX_SK_PHASE_ONE) {
            fprintf(stderr, "Warning: Number of phase 1 solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
            nSK1SolutionsCPU = MAX_SK_PHASE_ONE;
        }

        nBlocks = (nSK1SolutionsCPU + nThreads - 1) / nThreads;

        find_slide_kick_setupG2 << <nBlocks, nThreads >> > (devFloorPoints, nPoints, yNormal, platformMinX, platformMaxX, platformMinZ, platformMaxZ, midPointX, midPointZ);

        cudaMemcpyFromSymbol(&nSK2ASolutionsCPU, nSK2ASolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&nSK2BSolutionsCPU, nSK2BSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&nSK2CSolutionsCPU, nSK2CSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&nSK2DSolutionsCPU, nSK2DSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
    }

    if (nSK2ASolutionsCPU > 0) {
        if (nSK2ASolutionsCPU > MAX_SK_PHASE_TWO_A) {
            fprintf(stderr, "Warning: Number of phase 2a solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
            nSK2ASolutionsCPU = MAX_SK_PHASE_TWO_A;
        }

        nBlocks = (nSK2ASolutionsCPU + nThreads - 1) / nThreads;

        find_slide_kick_setupG3a << <nBlocks, nThreads >> > (platformMinZ, platformMaxZ);
    }

    if (nSK2BSolutionsCPU > 0) {
        if (nSK2BSolutionsCPU > MAX_SK_PHASE_TWO_B) {
            fprintf(stderr, "Warning: Number of phase 2b solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
            nSK2BSolutionsCPU = MAX_SK_PHASE_TWO_B;
        }

        nBlocks = (nSK2BSolutionsCPU + nThreads - 1) / nThreads;

        find_slide_kick_setupG3b << <nBlocks, nThreads >> > (platformMinX, platformMaxX);
    }

    if (nSK2CSolutionsCPU > 0) {
        if (nSK2CSolutionsCPU > MAX_SK_PHASE_TWO_C) {
            fprintf(stderr, "Warning: Number of phase 2c solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
            nSK2CSolutionsCPU = MAX_SK_PHASE_TWO_C;
        }

        nBlocks = (nSK2CSolutionsCPU + nThreads - 1) / nThreads;

        find_slide_kick_setupG3c << <nBlocks, nThreads >> > (platformMinX, platformMaxX, platformMinZ, platformMaxZ);
    }

    if (nSK2DSolutionsCPU > 0) {
        if (nSK2DSolutionsCPU > MAX_SK_PHASE_TWO_D) {
            fprintf(stderr, "Warning: Number of phase 2d solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
            nSK2DSolutionsCPU = MAX_SK_PHASE_TWO_D;
        }

        nBlocks = (nSK2DSolutionsCPU + nThreads - 1) / nThreads;

        find_slide_kick_setupG3d << <nBlocks, nThreads >> > (platformMinX, platformMaxX, platformMinZ, platformMaxZ);
    }

    cudaMemcpyFromSymbol(&nSK3SolutionsCPU, nSK3Solutions, sizeof(int), 0, cudaMemcpyDeviceToHost);

    if (nSK3SolutionsCPU > 0) {
        if (nSK3SolutionsCPU > MAX_SK_PHASE_THREE) {
            fprintf(stderr, "Warning: Number of phase 3 solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
            nSK3SolutionsCPU = MAX_SK_PHASE_THREE;
        }

        nBlocks = (nSK3SolutionsCPU + nThreads - 1) / nThreads;

        try_slide_kick_routeG << <nBlocks, nThreads >> > (devFloorPoints, nPoints);

        cudaMemcpyFromSymbol(&nSK4SolutionsCPU, nSK4Solutions, sizeof(int), 0, cudaMemcpyDeviceToHost);

    }

    if (nSK4SolutionsCPU > 0) {
        if (nSK4SolutionsCPU > MAX_SK_PHASE_FOUR) {
            fprintf(stderr, "Warning: Number of phase 4 solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
            nSK4SolutionsCPU = MAX_SK_PHASE_FOUR;
        }

        nBlocks = (nSK4SolutionsCPU + nThreads - 1) / nThreads;

        try_slide_kick_routeG2 << <nBlocks, nThreads >> > ();

        cudaMemcpyFromSymbol(&nSK5SolutionsCPU, nSK5Solutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
    }

    if (nSK5SolutionsCPU > 0) {
        if (nSK5SolutionsCPU > MAX_SK_PHASE_FIVE) {
            fprintf(stderr, "Warning: Number of phase 5 solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
            nSK5SolutionsCPU = MAX_SK_PHASE_FIVE;
        }

        nBlocks = (nSK5SolutionsCPU + nThreads - 1) / nThreads;

        try_stick_positionG << <nBlocks, nThreads >> > ();
    }
}


// Post-Platform/Upwarp Functions

__device__ float find_pre10K_speed(float post10KSpeed, float& post10KVelX, float& post10KVelZ, int solIdx) {
    struct SKPhase6* sol6 = &(sk6Solutions[solIdx]);
    struct SKPhase5* sol5 = &(sk5Solutions[sol6->p5Idx]);
    struct SKPhase4* sol4 = &(sk4Solutions[sol5->p4Idx]);
    struct SKPhase3* sol3 = &(sk3Solutions[sol4->p3Idx]);
    struct SKPhase2* sol2 = (sol3->p2Type / 2 == 0) ? ((sol3->p2Type % 2 == 0) ? &(sk2ASolutions[sol3->p2Idx]) : &(sk2BSolutions[sol3->p2Idx])) : ((sol3->p2Type % 2 == 0) ? &(sk2CSolutions[sol3->p2Idx]) : &(sk2DSolutions[sol3->p2Idx]));

    float pre10KSpeed = NAN;
    post10KVelX = NAN;
    post10KVelZ = NAN;

    int trueX = (sol5->stickX == 0) ? 0 : ((sol5->stickX < 0) ? sol5->stickX - 6 : sol5->stickX + 6);
    int trueY = (sol5->stickY == 0) ? 0 : ((sol5->stickY < 0) ? sol5->stickY - 6 : sol5->stickY + 6);

    float mag = sqrtf((float)(sol5->stickX * sol5->stickX + sol5->stickY * sol5->stickY));

    float xS = sol5->stickX;
    float yS = sol5->stickY;

    if (mag > 64.0f) {
        xS = xS * (64.0f / mag);
        yS = yS * (64.0f / mag);
        mag = 64.0f;
    }

    float intendedMag = ((mag / 64.0f) * (mag / 64.0f)) * 32.0f;
    int intendedYaw = atan2sG(-yS, xS) + sol4->cameraYaw;
    int intendedDYaw = intendedYaw - sol5->f1Angle;
    intendedDYaw = (65536 + (intendedDYaw % 65536)) % 65536;

    double w = intendedMag * gCosineTableG[intendedDYaw >> 4];
    double eqB = (50.0 + 147200.0 / w);
    double eqC = -(320000.0 / w) * post10KSpeed;
    double eqDet = eqB * eqB - eqC;

    if (eqDet >= 0) {
        pre10KSpeed = sqrt(eqDet) - eqB;

        if (pre10KSpeed >= 0) {
            bool searchLoop = true;
            bool speedTest = true;

            float upperSpeed = 2.0f * pre10KSpeed;
            float lowerSpeed = 0.0f;

            while (searchLoop) {
                pre10KSpeed = fmaxf((upperSpeed + lowerSpeed) / 2.0f, nextafterf(lowerSpeed, INFINITY));

                float pre10KVelX = pre10KSpeed * gSineTableG[sol2->f2Angle >> 4];
                float pre10KVelZ = pre10KSpeed * gCosineTableG[sol2->f2Angle >> 4];

                post10KVelX = pre10KVelX;
                post10KVelZ = pre10KVelZ;

                float oldSpeed = sqrtf(post10KVelX * post10KVelX + post10KVelZ * post10KVelZ);

                post10KVelX += post10KVelZ * (intendedMag / 32.0f) * gSineTableG[intendedDYaw >> 4] * 0.05f;
                post10KVelZ -= post10KVelX * (intendedMag / 32.0f) * gSineTableG[intendedDYaw >> 4] * 0.05f;

                float newSpeed = sqrtf(post10KVelX * post10KVelX + post10KVelZ * post10KVelZ);

                post10KVelX = post10KVelX * oldSpeed / newSpeed;
                post10KVelZ = post10KVelZ * oldSpeed / newSpeed;

                post10KVelX += 7.0f * tenKFloors[sol2->tenKFloorIdx][6];
                post10KVelZ += 7.0f * tenKFloors[sol2->tenKFloorIdx][8];

                float forward = gCosineTableG[intendedDYaw >> 4] * (0.5f + 0.5f * pre10KSpeed / 100.0f);
                float lossFactor = intendedMag / 32.0f * forward * 0.02f + 0.92f;

                post10KVelX *= lossFactor;
                post10KVelZ *= lossFactor;

                float post10KSpeedTest = -sqrtf(post10KVelX * post10KVelX + post10KVelZ * post10KVelZ);

                if (post10KSpeedTest == post10KSpeed) {
                    searchLoop = false;
                }
                else {
                    if (post10KSpeedTest < post10KSpeed) {
                        upperSpeed = pre10KSpeed;
                    }
                    else {
                        lowerSpeed = pre10KSpeed;
                    }

                    if (nextafterf(lowerSpeed, INFINITY) == upperSpeed) {
                        searchLoop = false;
                        pre10KSpeed = NAN;
                        post10KVelX = NAN;
                        post10KVelZ = NAN;
                    }
                }
            }
        }
    }

    return pre10KSpeed;
}

__global__ void test_speed_solution(short* floorPoints, bool* squishEdges, const int nPoints, float floorNormalY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nSpeedSolutionsSK, MAX_SPEED_SOLUTIONS_SK)) {
        struct SpeedSolutionSK* sol = &(speedSolutionsSK[idx]);
        struct SKUpwarpSolution* skuwSol = &(skuwSolutions[sol->skuwSolutionIdx]);
        struct UpwarpSolution* uwSol = &(upwarpSolutions[skuwSol->uwIdx]);
        struct PlatformSolution* platSol = &(platSolutions[uwSol->platformSolutionIdx]);
        struct SKPhase6* sol6 = &(sk6Solutions[skuwSol->skIdx]);
        struct SKPhase5* sol5 = &(sk5Solutions[sol6->p5Idx]);
        struct SKPhase4* sol4 = &(sk4Solutions[sol5->p4Idx]);
        struct SKPhase3* sol3 = &(sk3Solutions[sol4->p3Idx]);
        struct SKPhase2* sol2 = (sol3->p2Type / 2 == 0) ? ((sol3->p2Type % 2 == 0) ? &(sk2ASolutions[sol3->p2Idx]) : &(sk2BSolutions[sol3->p2Idx])) : ((sol3->p2Type % 2 == 0) ? &(sk2CSolutions[sol3->p2Idx]) : &(sk2DSolutions[sol3->p2Idx]));
        struct SKPhase1* sol1 = &(sk1Solutions[sol2->p1Idx]);

        float returnVelX;
        float returnVelZ;
        float pre10KSpeed = find_pre10K_speed(sol->returnSpeed, returnVelX, returnVelZ, skuwSol->skIdx);

        if (pre10KSpeed) {
            float frame2Position[3] = { platSol->returnPosition[0] - tenKFloors[sol2->tenKFloorIdx][7] * (returnVelX / 4.0f), platSol->returnPosition[1], platSol->returnPosition[2] - tenKFloors[sol2->tenKFloorIdx][7] * (returnVelZ / 4.0f) };

            SurfaceG* floor;
            float floorHeight;

            int floorIdx = find_floor(frame2Position, &floor, floorHeight, floorsG, total_floorsG);

            if (floorIdx != -1 && floor->normal[1] == tenKFloors[sol2->tenKFloorIdx][7] && floorHeight < platSol->returnPosition[1] && floorHeight >= platSol->returnPosition[1] - 78.0f && floorHeight > -2971.0f) {
                int returnSlideYaw = atan2sG(returnVelZ, returnVelX);
                int newFacingDYaw = (short)(sol2->f2Angle - returnSlideYaw);

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

                float postReturnVelX = sol->returnSpeed * gSineTableG[returnFaceAngle >> 4];
                float postReturnVelZ = sol->returnSpeed * gCosineTableG[returnFaceAngle >> 4];

                float intendedPosition[3] = { platSol->returnPosition[0] + postReturnVelX / 4.0, platSol->returnPosition[1], platSol->returnPosition[2] + postReturnVelZ / 4.0 };

                bool outOfBoundsTest = !check_inbounds(intendedPosition);

                for (int f = 0; outOfBoundsTest && f < 3; f++) {
                    intendedPosition[0] = platSol->landingPositions[f][0] + platSol->landingFloorNormalsY[f] * (postReturnVelX / 4.0);
                    intendedPosition[1] = platSol->landingPositions[f][1];
                    intendedPosition[2] = platSol->landingPositions[f][2] + platSol->landingFloorNormalsY[f] * (postReturnVelZ / 4.0);

                    outOfBoundsTest = !check_inbounds(intendedPosition);
                }

                if (outOfBoundsTest) {
                    frame2Position[1] = floorHeight;

                    float startSpeed = pre10KSpeed + 1.0f;
                    startSpeed = startSpeed + 0.35f;

                    float startVelX = pre10KSpeed * gSineTableG[sol2->f2Angle >> 4];
                    float startVelZ = pre10KSpeed * gCosineTableG[sol2->f2Angle >> 4];

                    float frame1Position[3] = { frame2Position[0], frame2Position[1], frame2Position[2] };

                    bool inBoundsTest = true;

                    for (int q = 0; q < sol1->q2; q++) {
                        frame1Position[0] = frame1Position[0] - (startVelX / 4.0f);
                        frame1Position[2] = frame1Position[2] - (startVelZ / 4.0f);

                        if (!check_inbounds(frame1Position)) {
                            inBoundsTest = false;
                            break;
                        }
                    }

                    if (inBoundsTest) {
                        floorIdx = find_floor_triangles(frame1Position, startTriangles, startNormals, &floorHeight);

                        if (floorIdx != -1 && floorHeight + (sol1->q2 * 20.0f / 4.0f) < frame2Position[1] && floorHeight + (sol1->q2 * 20.0f / 4.0f) >= frame2Position[1] - 78.0f && floorHeight > -3071.0f) {
                            frame1Position[1] = floorHeight;

                            float startPositions[2][3];
                            int intersections = 0;

                            for (int i = 0; i < nPoints; i++) {
                                if (squishEdges[i]) {
                                    double eqA = ((double)floorPoints[3 * ((i + 1) % nPoints)] - (double)floorPoints[3 * i]) * ((double)floorPoints[3 * ((i + 1) % nPoints)] - (double)floorPoints[3 * i]) + ((double)floorPoints[3 * ((i + 1) % nPoints) + 2] - (double)floorPoints[3 * i + 2]) * ((double)floorPoints[3 * ((i + 1) % nPoints) + 2] - (double)floorPoints[3 * i + 2]);
                                    double eqB = 2.0 * (((double)floorPoints[3 * ((i + 1) % nPoints)] - (double)floorPoints[3 * i]) * ((double)floorPoints[3 * i] - frame1Position[0]) + ((double)floorPoints[3 * ((i + 1) % nPoints) + 2] - (double)floorPoints[3 * i + 2]) * ((double)floorPoints[3 * i + 2] - frame1Position[2]));
                                    double eqC = ((double)floorPoints[3 * i] - frame1Position[0]) * ((double)floorPoints[3 * i] - frame1Position[0]) + ((double)floorPoints[3 * i + 2] - frame1Position[2]) * ((double)floorPoints[3 * i + 2] - frame1Position[2]) - ((double)startSpeed * (double)floorNormalY) * ((double)startSpeed * (double)floorNormalY);
                                    double eqDet = eqB * eqB - 4.0 * eqA * eqC;

                                    if (eqDet >= 0) {
                                        double t = (-eqB + sqrt(eqDet)) / (2.0 * eqA);

                                        if (t >= 0.0 && t <= 1.0) {
                                            startPositions[intersections][0] = ((double)floorPoints[3 * ((i + 1) % nPoints)] - (double)floorPoints[3 * i]) * t + (double)floorPoints[3 * i];
                                            startPositions[intersections][1] = ((double)floorPoints[3 * ((i + 1) % nPoints) + 1] - (double)floorPoints[3 * i + 1]) * t + (double)floorPoints[3 * i + 1];
                                            startPositions[intersections][2] = ((double)floorPoints[3 * ((i + 1) % nPoints) + 2] - (double)floorPoints[3 * i + 2]) * t + (double)floorPoints[3 * i + 2];
                                            intersections++;
                                            continue;
                                        }

                                        t = (-eqB - sqrt(eqDet)) / (2.0 * eqA);

                                        if (t >= 0.0 && t <= 1.0) {
                                            startPositions[intersections][0] = ((double)floorPoints[3 * ((i + 1) % nPoints)] - (double)floorPoints[3 * i]) * t + (double)floorPoints[3 * i];
                                            startPositions[intersections][1] = ((double)floorPoints[3 * ((i + 1) % nPoints) + 1] - (double)floorPoints[3 * i + 1]) * t + (double)floorPoints[3 * i + 1];
                                            startPositions[intersections][2] = ((double)floorPoints[3 * ((i + 1) % nPoints) + 2] - (double)floorPoints[3 * i + 2]) * t + (double)floorPoints[3 * i + 2];
                                            intersections++;
                                        }
                                    }
                                }
                            }

                            for (int i = 0; i < intersections; i++) {
                                if (startPositions[i][1] > -2971.0f && startPositions[i][1] < -2921.0f - (52.0f * sqrtf(1.0f - floorNormalY * floorNormalY) / floorNormalY)) {
                                    int f1Angle = atan2sG(frame1Position[2] - startPositions[i][2], frame1Position[0] - startPositions[i][0]);
                                    f1Angle = (65536 + f1Angle) % 65536;

                                    if (f1Angle == sol5->f1Angle) {
                                        int solIdx = atomicAdd(&n10KSolutionsSK, 1);

                                        if (solIdx < MAX_SK_UPWARP_SOLUTIONS) {
                                            struct TenKSolutionSK* solution = &(tenKSolutionsSK[solIdx]);
                                            solution->speedSolutionIdx = idx;
                                            solution->pre10KSpeed = startSpeed;
                                            solution->pre10KVel[0] = startVelX;
                                            solution->pre10KVel[1] = startVelZ;
                                            solution->returnVel[0] = returnVelX;
                                            solution->returnVel[1] = returnVelZ;
                                            solution->frame2Position[0] = frame2Position[0];
                                            solution->frame2Position[1] = frame2Position[1];
                                            solution->frame2Position[2] = frame2Position[2];
                                            solution->frame1Position[0] = frame1Position[0];
                                            solution->frame1Position[1] = frame1Position[1];
                                            solution->frame1Position[2] = frame1Position[2];
                                            solution->startPosition[0] = startPositions[i][0];
                                            solution->startPosition[1] = startPositions[i][1];
                                            solution->startPosition[2] = startPositions[i][2];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

__global__ void find_speed_solutions() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nSKUWSolutions, MAX_SK_UPWARP_SOLUTIONS)) {
        struct SKUpwarpSolution* sol = &(skuwSolutions[idx]);
        struct UpwarpSolution* uwSol = &(upwarpSolutions[sol->uwIdx]);
        struct PlatformSolution* platSol = &(platSolutions[uwSol->platformSolutionIdx]);

        struct SKPhase6* sol6 = &(sk6Solutions[sol->skIdx]);
        struct SKPhase5* sol5 = &(sk5Solutions[sol6->p5Idx]);
        struct SKPhase4* sol4 = &(sk4Solutions[sol5->p4Idx]);
        struct SKPhase3* sol3 = &(sk3Solutions[sol4->p3Idx]);
        struct SKPhase2* sol2 = (sol3->p2Type / 2 == 0) ? ((sol3->p2Type % 2 == 0) ? &(sk2ASolutions[sol3->p2Idx]) : &(sk2BSolutions[sol3->p2Idx])) : ((sol3->p2Type % 2 == 0) ? &(sk2CSolutions[sol3->p2Idx]) : &(sk2DSolutions[sol3->p2Idx]));
        struct SKPhase1* sol1 = &(sk1Solutions[sol2->p1Idx]);

        float minX = 65536.0f * sol3->x2 + tenKFloors[sol2->tenKFloorIdx][0];
        float maxX = 65536.0f * sol3->x2 + tenKFloors[sol2->tenKFloorIdx][1];
        float minZ = 65536.0f * sol3->z2 + tenKFloors[sol2->tenKFloorIdx][2];
        float maxZ = 65536.0f * sol3->z2 + tenKFloors[sol2->tenKFloorIdx][3];

        float minSpeed = sol->minSpeed;
        float maxSpeed = sol->maxSpeed;

        float minReturnVelX;
        float minReturnVelZ;
        float minPre10KSpeed = NAN;

        while (!minPre10KSpeed && minSpeed >= maxSpeed) {
            minPre10KSpeed = find_pre10K_speed(minSpeed, minReturnVelX, minReturnVelZ, sol->skIdx);

            if (!minPre10KSpeed) {
                minSpeed = nextafterf(minSpeed, -INFINITY);
            }
        }

        float maxReturnVelX;
        float maxReturnVelZ;
        float maxPre10KSpeed = NAN;

        while (!maxPre10KSpeed && maxSpeed <= minSpeed) {
            maxPre10KSpeed = find_pre10K_speed(maxSpeed, maxReturnVelX, maxReturnVelZ, sol->skIdx);

            if (!maxPre10KSpeed) {
                maxSpeed = nextafterf(maxSpeed, INFINITY);
            }
        }

        if (minSpeed >= maxSpeed) {
            float minSpeedF2X = platSol->returnPosition[0] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelX / 4.0);
            float minSpeedF2Z = platSol->returnPosition[2] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelZ / 4.0);

            float maxSpeedF2X = platSol->returnPosition[0] - tenKFloors[sol2->tenKFloorIdx][7] * (maxReturnVelX / 4.0);
            float maxSpeedF2Z = platSol->returnPosition[2] - tenKFloors[sol2->tenKFloorIdx][7] * (maxReturnVelZ / 4.0);

            bool speedTest = true;

            if (minSpeedF2X < minX) {
                if (maxSpeedF2X < minX) {
                    speedTest = false;
                }
                else {
                    float lowerSpeed = minSpeed;
                    float upperSpeed = maxSpeed;

                    while (nextafter(lowerSpeed, -INFINITY) > upperSpeed) {
                        float midSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));

                        minPre10KSpeed = find_pre10K_speed(midSpeed, minReturnVelX, minReturnVelZ, sol->skIdx);

                        if (minPre10KSpeed) {
                            minSpeedF2X = platSol->returnPosition[0] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelX / 4.0);

                            if (minSpeedF2X < minX) {
                                lowerSpeed = midSpeed;
                            }
                            else {
                                upperSpeed = midSpeed;
                            }
                        }
                    }

                    minSpeed = upperSpeed;
                }
            }
            else if (maxSpeedF2X < maxX) {
                float lowerSpeed = minSpeed;
                float upperSpeed = maxSpeed;

                while (nextafter(lowerSpeed, -INFINITY) > upperSpeed) {
                    float midSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));

                    maxPre10KSpeed = find_pre10K_speed(midSpeed, maxReturnVelX, maxReturnVelZ, sol->skIdx);

                    if (maxPre10KSpeed) {
                        maxSpeedF2X = platSol->returnPosition[0] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelX / 4.0);

                        if (maxSpeedF2X < minX) {
                            upperSpeed = midSpeed;
                        }
                        else {
                            lowerSpeed = midSpeed;
                        }
                    }
                }

                maxSpeed = lowerSpeed;
            }

            if (speedTest) {
                if (minSpeedF2X > maxX) {
                    if (maxSpeedF2X > maxX) {
                        speedTest = false;
                    }
                    else {
                        float lowerSpeed = minSpeed;
                        float upperSpeed = maxSpeed;

                        while (nextafter(lowerSpeed, -INFINITY) > upperSpeed) {
                            float midSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));

                            minPre10KSpeed = find_pre10K_speed(midSpeed, minReturnVelX, minReturnVelZ, sol->skIdx);

                            if (minPre10KSpeed) {
                                minSpeedF2X = platSol->returnPosition[0] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelX / 4.0);

                                if (minSpeedF2X > maxX) {
                                    lowerSpeed = midSpeed;
                                }
                                else {
                                    upperSpeed = midSpeed;
                                }
                            }
                        }

                        minSpeed = upperSpeed;
                    }
                }
                else if (maxSpeedF2X > maxX) {
                    float lowerSpeed = minSpeed;
                    float upperSpeed = maxSpeed;

                    while (nextafter(lowerSpeed, -INFINITY) > upperSpeed) {
                        float midSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));

                        maxPre10KSpeed = find_pre10K_speed(midSpeed, maxReturnVelX, maxReturnVelZ, sol->skIdx);

                        if (maxPre10KSpeed) {
                            maxSpeedF2X = platSol->returnPosition[0] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelX / 4.0);

                            if (maxSpeedF2X > maxX) {
                                upperSpeed = midSpeed;
                            }
                            else {
                                lowerSpeed = midSpeed;
                            }
                        }
                    }

                    minSpeed = lowerSpeed;
                }
            }

            if (speedTest) {
                if (minSpeedF2Z < minZ) {
                    if (maxSpeedF2Z < minZ) {
                        speedTest = false;
                    }
                    else {
                        float lowerSpeed = minSpeed;
                        float upperSpeed = maxSpeed;

                        while (nextafter(lowerSpeed, -INFINITY) > upperSpeed) {
                            float midSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));

                            minPre10KSpeed = find_pre10K_speed(midSpeed, minReturnVelX, minReturnVelZ, sol->skIdx);

                            if (minPre10KSpeed) {
                                minSpeedF2Z = platSol->returnPosition[2] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelZ / 4.0);

                                if (minSpeedF2Z < minZ) {
                                    lowerSpeed = midSpeed;
                                }
                                else {
                                    upperSpeed = midSpeed;
                                }
                            }
                        }

                        minSpeed = upperSpeed;
                    }
                }
                else if (maxSpeedF2Z < maxZ) {
                    float lowerSpeed = minSpeed;
                    float upperSpeed = maxSpeed;

                    while (nextafter(lowerSpeed, -INFINITY) > upperSpeed) {
                        float midSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));

                        maxPre10KSpeed = find_pre10K_speed(midSpeed, maxReturnVelX, maxReturnVelZ, sol->skIdx);

                        if (maxPre10KSpeed) {
                            maxSpeedF2Z = platSol->returnPosition[2] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelZ / 4.0);

                            if (maxSpeedF2Z < minZ) {
                                upperSpeed = midSpeed;
                            }
                            else {
                                lowerSpeed = midSpeed;
                            }
                        }
                    }

                    maxSpeed = lowerSpeed;
                }
            }

            if (speedTest) {
                if (minSpeedF2Z > maxZ) {
                    if (maxSpeedF2Z > maxZ) {
                        speedTest = false;
                    }
                    else {
                        float lowerSpeed = minSpeed;
                        float upperSpeed = maxSpeed;

                        while (nextafter(lowerSpeed, -INFINITY) > upperSpeed) {
                            float midSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));

                            minPre10KSpeed = find_pre10K_speed(midSpeed, minReturnVelX, minReturnVelZ, sol->skIdx);

                            if (minPre10KSpeed) {
                                minSpeedF2Z = platSol->returnPosition[2] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelZ / 4.0);

                                if (minSpeedF2Z > maxZ) {
                                    lowerSpeed = midSpeed;
                                }
                                else {
                                    upperSpeed = midSpeed;
                                }
                            }
                        }

                        minSpeed = upperSpeed;
                    }
                }
                else if (maxSpeedF2Z > maxZ) {
                    float lowerSpeed = minSpeed;
                    float upperSpeed = maxSpeed;

                    while (nextafter(lowerSpeed, -INFINITY) > upperSpeed) {
                        float midSpeed = fminf((lowerSpeed + upperSpeed) / 2.0, nextafter(lowerSpeed, -INFINITY));

                        maxPre10KSpeed = find_pre10K_speed(midSpeed, maxReturnVelX, maxReturnVelZ, sol->skIdx);

                        if (maxPre10KSpeed) {
                            maxSpeedF2Z = platSol->returnPosition[2] - tenKFloors[sol2->tenKFloorIdx][7] * (minReturnVelZ / 4.0);

                            if (maxSpeedF2Z > maxZ) {
                                upperSpeed = midSpeed;
                            }
                            else {
                                lowerSpeed = midSpeed;
                            }
                        }
                    }

                    minSpeed = lowerSpeed;
                }
            }

            if (speedTest) {
                for (float speed = minSpeed; speed >= maxSpeed; speed = nextafterf(speed, -INFINITY)) {
                    int solIdx = atomicAdd(&nSpeedSolutionsSK, 1);

                    if (solIdx < MAX_SPEED_SOLUTIONS_SK) {
                        struct SpeedSolutionSK* solution = &(speedSolutionsSK[solIdx]);
                        solution->skuwSolutionIdx = idx;
                        solution->returnSpeed = speed;
                    }
                }
            }
        }
    }
}

__global__ void find_sk_upwarp_solutions() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nUpwarpSolutions, MAX_UPWARP_SOLUTIONS)) {
        struct UpwarpSolution* uwSol = &(upwarpSolutions[idx]);
        struct PlatformSolution* platSol = &(platSolutions[uwSol->platformSolutionIdx]);

        float speedBuffer = 1000.0;

        double maxDist = -INFINITY;
        double minDist = INFINITY;

        for (int i = 0; i < 3; i++) {
            double xDist = 65536.0f * uwSol->pux + platSol->endTriangles[platSol->endFloorIdx][i][0] - platSol->returnPosition[0];
            double zDist = 65536.0f * uwSol->puz + platSol->endTriangles[platSol->endFloorIdx][i][2] - platSol->returnPosition[2];

            double dist = sqrt(xDist * xDist + zDist * zDist);

            minDist = fmin(dist, minDist);
            maxDist = fmax(dist, maxDist);
        }

        float upperSpeed = -(maxDist / platSol->endTriangleNormals[platSol->endFloorIdx][1]) / 0.9 - speedBuffer;
        float lowerSpeed = -(minDist / platSol->endTriangleNormals[platSol->endFloorIdx][1]) / 0.94 + speedBuffer;

        for (int i = 0; i < min(nSK6Solutions, MAX_SK_PHASE_SIX); i++) {
            SKPhase6* sk6Sol = &(sk6Solutions[i]);
            struct SKPhase5* sol5 = &(sk5Solutions[sk6Sol->p5Idx]);
            struct SKPhase4* sol4 = &(sk4Solutions[sol5->p4Idx]);
            struct SKPhase3* sol3 = &(sk3Solutions[sol4->p3Idx]);
            struct SKPhase2* sol2 = (sol3->p2Type / 2 == 0) ? ((sol3->p2Type % 2 == 0) ? &(sk2ASolutions[sol3->p2Idx]) : &(sk2BSolutions[sol3->p2Idx])) : ((sol3->p2Type % 2 == 0) ? &(sk2CSolutions[sol3->p2Idx]) : &(sk2DSolutions[sol3->p2Idx]));
            struct SKPhase1* sol1 = &(sk1Solutions[sol2->p1Idx]);

            float minSpeed = fminf(lowerSpeed, sk6Sol->minPost10KSpeed);
            float maxSpeed = fmaxf(upperSpeed, sk6Sol->maxPost10KSpeed);

            if (minSpeed >= maxSpeed) {
                int solIdx = atomicAdd(&nSKUWSolutions, 1);

                if (solIdx < MAX_SK_UPWARP_SOLUTIONS) {
                    struct SKUpwarpSolution* solution = &(skuwSolutions[solIdx]);
                    solution->skIdx = i;
                    solution->uwIdx = idx;
                    solution->minSpeed = minSpeed;
                    solution->maxSpeed = maxSpeed;
                }
            }
        }
    }
}

__device__ void try_upwarp_slide(int solIdx, int angle, int intendedDYaw, float intendedMag) {
    struct TenKSolutionSK* tenKSol = &(tenKSolutionsSK[solIdx]);
    struct SpeedSolutionSK* speedSol = &(speedSolutionsSK[tenKSol->speedSolutionIdx]);
    struct SKUpwarpSolution* skuwSol = &(skuwSolutions[speedSol->skuwSolutionIdx]);
    struct UpwarpSolution* uwSol = &(upwarpSolutions[skuwSol->uwIdx]);
    struct PlatformSolution* platSol = &(platSolutions[uwSol->platformSolutionIdx]);

    float lossFactor = intendedMag / 32.0f * gCosineTableG[intendedDYaw >> 4] * 0.02f + 0.92f;
    int slopeAngle = atan2sG(platSol->endTriangleNormals[platSol->endFloorIdx][2], platSol->endTriangleNormals[platSol->endFloorIdx][0]);
    float steepness = sqrtf(platSol->endTriangleNormals[platSol->endFloorIdx][0] * platSol->endTriangleNormals[platSol->endFloorIdx][0] + platSol->endTriangleNormals[platSol->endFloorIdx][2] * platSol->endTriangleNormals[platSol->endFloorIdx][2]);

    float xVel0 = speedSol->returnSpeed * gSineTableG[angle >> 4];
    float zVel0 = speedSol->returnSpeed * gCosineTableG[angle >> 4];

    float xVel1 = xVel0;
    float zVel1 = zVel0;

    float oldSpeed = sqrtf(xVel1 * xVel1 + zVel1 * zVel1);

    xVel1 += zVel1 * (intendedMag / 32.0f) * gSineTableG[intendedDYaw >> 4] * 0.05f;
    zVel1 -= xVel1 * (intendedMag / 32.0f) * gSineTableG[intendedDYaw >> 4] * 0.05f;

    float newSpeed = sqrtf(xVel1 * xVel1 + zVel1 * zVel1);

    xVel1 = xVel1 * oldSpeed / newSpeed;
    zVel1 = zVel1 * oldSpeed / newSpeed;

    xVel1 += 7.0f * steepness * gSineTableG[slopeAngle >> 4];
    zVel1 += 7.0f * steepness * gCosineTableG[slopeAngle >> 4];

    xVel1 *= lossFactor;
    zVel1 *= lossFactor;

    float intendedPos[3] = { platSol->endPosition[0], platSol->endPosition[1], platSol->endPosition[2] };

    int floorIdx = platSol->endFloorIdx;
    bool slideCheck = true;

    for (int s = 0; s < 4; s++) {
        intendedPos[0] = intendedPos[0] + platSol->endTriangleNormals[floorIdx][1] * (xVel1 / 4.0f);
        intendedPos[2] = intendedPos[2] + platSol->endTriangleNormals[floorIdx][1] * (zVel1 / 4.0f);

        float floorHeight;
        floorIdx = find_floor_triangles(intendedPos, platSol->endTriangles, platSol->endTriangleNormals, &floorHeight);

        if (floorIdx == -1 || floorHeight <= -3071.0f) {
            slideCheck = false;
            break;
        }
        else {
            intendedPos[1] = floorHeight;
        }
    }

    if (slideCheck) {
        float prePositionTest[3] = { platSol->penultimatePosition[0] + platSol->penultimateFloorNormalY * xVel0 / 4.0f, platSol->penultimatePosition[1], platSol->penultimatePosition[2] + platSol->penultimateFloorNormalY * zVel0 / 4.0f };

        if (!check_inbounds(prePositionTest)) {
            float test_normal[3] = { platSol->endNormal[0], platSol->endNormal[1], platSol->endNormal[2] };
            float mario_pos[3] = { intendedPos[0], intendedPos[1], intendedPos[2] };

            short triangles[2][3][3];
            float normals[2][3];
            float mat[4][4];

            platform_logic_gpu(test_normal, mario_pos, triangles, normals, mat);

            bool upwarpPositionTest = false;

            for (int i = 0; i < n_y_ranges && !upwarpPositionTest; i++) {
                if (mario_pos[1] >= lower_y[i] && mario_pos[1] <= upper_y[i]) {
                    upwarpPositionTest = true;
                }
            }

            upwarpPositionTest = upwarpPositionTest && check_inbounds(mario_pos);

            if (upwarpPositionTest) {
                int idx = atomicAdd(&nSlideSolutions, 1);
                if (idx < MAX_SLIDE_SOLUTIONS) {
                    int slideYaw = atan2sG(zVel1, xVel1);
                    slideYaw = (65536 + slideYaw) % 65536;

                    int facingDYaw = angle - slideYaw;

                    int newFacingDYaw = (short)facingDYaw;

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

                    int postSlideAngle = slideYaw + newFacingDYaw;
                    postSlideAngle = (65536 + postSlideAngle) % 65536;

                    float postSlideSpeed = -sqrtf(xVel1 * xVel1 + zVel1 * zVel1);

                    SlideSolution* solution = &(slideSolutions[idx]);
                    solution->tenKSolutionIdx = solIdx;
                    solution->preUpwarpPosition[0] = intendedPos[0];
                    solution->preUpwarpPosition[1] = intendedPos[1];
                    solution->preUpwarpPosition[2] = intendedPos[2];
                    solution->upwarpPosition[0] = mario_pos[0];
                    solution->upwarpPosition[1] = mario_pos[1];
                    solution->upwarpPosition[2] = mario_pos[2];
                    solution->angle = angle;
                    solution->intendedDYaw = intendedDYaw;
                    solution->stickMag = intendedMag;
                    solution->postSlideAngle = postSlideAngle;
                    solution->postSlideSpeed = postSlideSpeed;
                }
            }
        }
    }
}

__device__ void try_pu_slide_angle_sk(int solIdx, int angle, double minEndAngle, double maxEndAngle, double minM1, double maxM1) {
    double minAngleDiff = fmax(minEndAngle - angle, -(double)522);
    double maxAngleDiff = fmax(maxEndAngle - angle, (double)522);

    if (minAngleDiff <= maxAngleDiff) {
        double minEndAngleA = minAngleDiff + angle;
        double maxEndAngleA = maxAngleDiff + angle;

        double minN;
        double maxN;

        if (angle == 0 || angle == 32768) {
            double sinStartAngle = sin(2.0 * M_PI * (double)angle / 65536.0);

            minN = -cos(2.0 * M_PI * minEndAngleA / 65536.0) / sinStartAngle;
            maxN = -cos(2.0 * M_PI * maxEndAngleA / 65536.0) / sinStartAngle;
        }
        else {
            double sinStartAngle = gSineTableG[angle >> 4];
            double cosStartAngle = gCosineTableG[angle >> 4];

            double sinMinEndAngle = sin(2.0 * M_PI * minEndAngleA / 65536.0);
            double cosMinEndAngle = cos(2.0 * M_PI * minEndAngleA / 65536.0);

            double sinMaxEndAngle = sin(2.0 * M_PI * maxEndAngleA / 65536.0);
            double cosMaxEndAngle = cos(2.0 * M_PI * maxEndAngleA / 65536.0);

            double t = sinStartAngle / cosStartAngle;
            double s = sinMinEndAngle / cosMinEndAngle;

            bool signTest = (cosStartAngle > 0 && cosMinEndAngle > 0) || (cosStartAngle < 0 && cosMinEndAngle < 0);

            if (signTest) {
                minN = (-((double)s * (double)t) - 1.0 + sqrt(((double)s * (double)t - 1.0) * ((double)s * (double)t - 1.0) + 4.0 * (double)s * (double)s)) / (2.0 * (double)s);
            }
            else {
                minN = (-((double)s * (double)t) - 1.0 - sqrt(((double)s * (double)t - 1.0) * ((double)s * (double)t - 1.0) + 4.0 * (double)s * (double)s)) / (2.0 * (double)s);
            }

            s = sinMaxEndAngle / cosMaxEndAngle;

            signTest = (cosStartAngle > 0 && cosMaxEndAngle > 0) || (cosStartAngle < 0 && cosMaxEndAngle < 0);

            if (signTest) {
                maxN = (-((double)s * (double)t) - 1.0 + sqrt(((double)s * (double)t - 1.0) * ((double)s * (double)t - 1.0) + 4.0 * (double)s * (double)s)) / (2.0 * (double)s);
            }
            else {
                maxN = (-((double)s * (double)t) - 1.0 - sqrt(((double)s * (double)t - 1.0) * ((double)s * (double)t - 1.0) + 4.0 * (double)s * (double)s)) / (2.0 * (double)s);
            }
        }

        double minN1 = 32.0 * minN / 0.05;
        double maxN1 = 32.0 * maxN / 0.05;

        if (minN1 > maxN1) {
            double temp = minN1;
            minN1 = maxN1;
            maxN1 = temp;
        }

        minN1 = fmax(minN1, -32.0);
        maxN1 = fmin(maxN1, 32.0);

        if (minN1 <= maxN1) {
            double minMag = INFINITY;
            double maxMag = -INFINITY;

            double minYaw = INFINITY;
            double maxYaw = -INFINITY;

            for (int i = 0; i < 4; i++) {
                double m1;
                double n1;

                if (i % 2 == 0) {
                    m1 = minM1;
                }
                else {
                    m1 = maxM1;
                }

                if (i / 2 == 0) {
                    n1 = minN1;
                }
                else {
                    n1 = maxN1;
                }

                double mag = sqrt(m1 * m1 + n1 * n1);
                double yaw = 65536.0 * (atan2(n1, m1) / (2.0 * M_PI));
                yaw = fmod(65536.0 + yaw, 65536.0);

                minMag = fmin(minMag, mag);
                maxMag = fmax(maxMag, mag);
                minYaw = fmin(minYaw, yaw - angle);
                maxYaw = fmax(maxYaw, yaw - angle);
            }

            maxMag = fmin(maxMag, 32.0);

            if (minMag <= maxMag) {
                int minIntendedDYaw = 16 * (int)ceil((minYaw + angle) / 16);
                int maxIntendedDYaw = 16 * (int)floor((maxYaw + angle) / 16);

                int minIdx = -1;
                int maxIdx = magCount;

                while (maxIdx > minIdx + 1) {
                    int midIdx = (maxIdx + minIdx) / 2;

                    if (minMag - 0.001 < magSet[midIdx]) {
                        maxIdx = midIdx;
                    }
                    else {
                        minIdx = midIdx;
                    }
                }

                int startMagIdx = maxIdx;

                minIdx = -1;
                maxIdx = magCount;

                while (maxIdx > minIdx + 1) {
                    int midIdx = (maxIdx + minIdx) / 2;

                    if (maxMag + 0.001 < magSet[midIdx]) {
                        maxIdx = midIdx;
                    }
                    else {
                        minIdx = midIdx;
                    }
                }

                int endMagIdx = minIdx;

                for (int intendedDYaw = minIntendedDYaw; intendedDYaw <= maxIntendedDYaw; intendedDYaw += 16) {
                    for (int magIdx = startMagIdx; magIdx <= endMagIdx; magIdx++) {
                        float intendedMag = magSet[magIdx];

                        try_upwarp_slide(solIdx, angle, intendedDYaw, intendedMag);
                    }
                }
            }
        }
    }
}

__global__ void find_slide_solutions() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(n10KSolutionsSK, MAX_10K_SOLUTIONS_SK)) {
        struct TenKSolutionSK* tenKSol = &(tenKSolutionsSK[idx]);
        struct SpeedSolutionSK* speedSol = &(speedSolutionsSK[tenKSol->speedSolutionIdx]);
        struct SKUpwarpSolution* skuwSol = &(skuwSolutions[speedSol->skuwSolutionIdx]);
        struct UpwarpSolution* uwSol = &(upwarpSolutions[skuwSol->uwIdx]);
        struct PlatformSolution* platSol = &(platSolutions[uwSol->platformSolutionIdx]);

        int maxTurnAngle = 522;

        double minDist = INFINITY;
        double maxDist = -INFINITY;

        double baseAngle = 0.0;

        double minEndAngle = 0.0;
        double maxEndAngle = 0.0;

        for (int i = 0; i < 3; i++) {
            double xDist = 65536.0 * uwSol->pux + platSol->endTriangles[platSol->endFloorIdx][i][0] - platSol->endPosition[0];
            double zDist = 65536.0 * uwSol->puz + platSol->endTriangles[platSol->endFloorIdx][i][2] - platSol->endPosition[2];

            double dist = sqrt(xDist * xDist + zDist * zDist);

            double angle = atan2(-xDist, -zDist);
            angle = fmod(2.0 * M_PI + angle, 2.0 * M_PI);

            if (i == 0) {
                baseAngle = angle;
            }
            else {
                minEndAngle = fmin(minEndAngle, angle - baseAngle);
                maxEndAngle = fmax(maxEndAngle, angle - baseAngle);
            }

            minDist = fmin(minDist, dist);
            maxDist = fmax(maxDist, dist);
        }

        double minSpeed = -minDist / (double)platSol->endTriangleNormals[platSol->endFloorIdx][1];
        double maxSpeed = -maxDist / (double)platSol->endTriangleNormals[platSol->endFloorIdx][1];

        double minM = minSpeed / (double)speedSol->returnSpeed;
        double maxM = maxSpeed / (double)speedSol->returnSpeed;

        double minM1 = 32.0 * ((minM - 0.92) / 0.02);
        double maxM1 = 32.0 * ((maxM - 0.92) / 0.02);

        if (minM1 > maxM1) {
            double temp = minM1;
            minM1 = maxM1;
            maxM1 = temp;
        }

        minM1 = fmax(minM1, -32.0);
        maxM1 = fmin(maxM1, 32.0);

        if (minM1 <= maxM1) {
            minEndAngle = minEndAngle + baseAngle;
            maxEndAngle = maxEndAngle + baseAngle;

            minEndAngle = 65536.0 * minEndAngle / (2.0 * M_PI);
            maxEndAngle = 65536.0 * maxEndAngle / (2.0 * M_PI);

            int minStartAngle = (int)ceil(minEndAngle) - maxTurnAngle;
            int maxStartAngle = (int)floor(maxEndAngle) + maxTurnAngle;

            minStartAngle = minStartAngle + 15;
            minStartAngle = minStartAngle - (minStartAngle % 16);

            for (int a = minStartAngle; a <= maxStartAngle; a += 16) {
                int angle = a % 65536;

                try_pu_slide_angle_sk(idx, angle, minEndAngle, maxEndAngle, minM1, maxM1);
            }
        }
    }
}

__global__ void find_breakdance_solutions() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nSlideSolutions, MAX_SLIDE_SOLUTIONS)) {
        struct SlideSolution* slideSol = &(slideSolutions[idx]);

        SurfaceG* floor;
        float floorHeight;

        int floorIdx = find_floor(slideSol->upwarpPosition, &floor, floorHeight, floorsG, total_floorsG);

        int slopeAngle = atan2sG(floor->normal[2], floor->normal[0]);
        slopeAngle = (slopeAngle + 65536) % 65536;

        float steepness = sqrtf(floor->normal[0] * floor->normal[0] + floor->normal[0] * floor->normal[0]);

        float cameraPositions[4][3] = { {-8192, -2918, -8192}, {-8192, -2918, 8191}, {8191, -2918, -8192}, {8191, -2918, 8191} };

        int minCameraYaw = 0;
        int maxCameraYaw = 0;

        int refCameraYaw = calculate_camera_yaw(slideSol->upwarpPosition, cameraPositions[0]);
        refCameraYaw = (65536 + refCameraYaw) % 65536;

        for (int i = 1; i < 4; i++) {
            int cameraYaw = calculate_camera_yaw(slideSol->upwarpPosition, cameraPositions[i]);
            cameraYaw = (short)(cameraYaw - refCameraYaw);
            minCameraYaw = min(minCameraYaw, cameraYaw);
            maxCameraYaw = max(maxCameraYaw, cameraYaw);
        }

        int minCameraIdx = gReverseArctanTable[(65536 + minCameraYaw + refCameraYaw) % 65536];
        int maxCameraIdx = gReverseArctanTable[(65536 + maxCameraYaw + refCameraYaw) % 65536];

        if (minCameraIdx > maxCameraIdx) {
            maxCameraIdx += 8192;
        }

        for (int i = minCameraIdx; i <= maxCameraIdx; i++) {
            int cameraYaw = gArctanTableG[i % 8192];
            cameraYaw = (65536 + cameraYaw) % 65536;

            if (validCameraAngle[cameraYaw]) {
                for (int x = -121; x <= 120; x++) {
                    for (int y = -121; y <= 120; y++) {
                        float stickX = x - (x < 0) + (x > 0);
                        float stickY = y - (y < 0) + (y > 0);

                        int rawX = x - 7 * (x < 0) + 7 * (x > 0);
                        int rawY = y - 7 * (y < 0) + 7 * (y > 0);

                        float stickMag = sqrtf(stickX * stickX + stickY * stickY);

                        if (stickMag > 64) {
                            stickX *= 64 / stickMag;
                            stickY *= 64 / stickMag;
                            stickMag = 64;
                        }

                        float intendedMag = ((stickMag / 64.0f) * (stickMag / 64.0f)) * 32.0f;

                        int intendedYaw;

                        if (intendedMag > 0.0f) {
                            intendedYaw = atan2sG(-stickY, stickX) + cameraYaw;
                            intendedYaw = (65536 + intendedYaw) % 65536;
                        }
                        else {
                            intendedYaw = slideSol->postSlideAngle;
                        }

                        int intendedDYaw = (short)(intendedYaw - slideSol->postSlideAngle);
                        intendedDYaw = (65536 + intendedDYaw) % 65536;

                        float lossFactor = intendedMag / 32.0f * gCosineTableG[intendedDYaw >> 4] * 0.02f + 0.92f;

                        float xVel = slideSol->postSlideSpeed * gSineTableG[slideSol->postSlideAngle >> 4];
                        float zVel = slideSol->postSlideSpeed * gCosineTableG[slideSol->postSlideAngle >> 4];

                        float oldSpeed = sqrtf(xVel * xVel + zVel * zVel);

                        xVel += zVel * (intendedMag / 32.0f) * gSineTableG[intendedDYaw >> 4] * 0.05f;
                        zVel -= xVel * (intendedMag / 32.0f) * gSineTableG[intendedDYaw >> 4] * 0.05f;

                        float newSpeed = sqrtf(xVel * xVel + zVel * zVel);

                        xVel = xVel * oldSpeed / newSpeed;
                        zVel = zVel * oldSpeed / newSpeed;

                        xVel += 7.0f * steepness * gSineTableG[slopeAngle >> 4];
                        zVel += 7.0f * steepness * gCosineTableG[slopeAngle >> 4];

                        xVel *= lossFactor;
                        zVel *= lossFactor;

                        float intendedPos[3] = { slideSol->upwarpPosition[0], slideSol->upwarpPosition[1], slideSol->upwarpPosition[2] };
                        SurfaceG* newFloor = floor;
                        bool fallTest = false;

                        for (int j = 0; j < 4; j++) {
                            intendedPos[0] = intendedPos[0] + newFloor->normal[1] * (xVel / 4.0f);
                            intendedPos[2] = intendedPos[2] + newFloor->normal[1] * (zVel / 4.0f);

                            int floorIdx = find_floor(intendedPos, &newFloor, floorHeight, floorsG, total_floorsG);

                            if (floorIdx == -1) {
                                break;
                            }
                            else if (intendedPos[1] > floorHeight + 100.0f) {
                                fallTest = true;
                                break;
                            }
                        }

                        if (fallTest) {
                            int slideYaw = atan2sG(zVel, xVel);
                            slideYaw = (65536 + slideYaw) % 65536;

                            int facingDYaw = slideSol->postSlideAngle - slideYaw;

                            int newFacingDYaw = (short)facingDYaw;

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

                            int postSlideAngle = slideYaw + newFacingDYaw;
                            postSlideAngle = (65536 + postSlideAngle) % 65536;

                            float postSlideSpeed = -sqrtf(xVel * xVel + zVel * zVel);

                            xVel = postSlideSpeed * gSineTableG[postSlideAngle >> 4];
                            float yVel = 0.0f;
                            zVel = postSlideSpeed * gCosineTableG[postSlideAngle >> 4];

                            bool falling = true;
                            bool landed = false;

                            while (falling) {
                                for (int j = 0; j < 4; j++) {
                                    intendedPos[0] = intendedPos[0] + (xVel / 4.0f);
                                    intendedPos[1] = intendedPos[1] + (yVel / 4.0f);
                                    intendedPos[2] = intendedPos[2] + (zVel / 4.0f);

                                    float oldFloorHeight = floorHeight;
                                    int floorIdx = find_floor(intendedPos, &newFloor, floorHeight, floorsG, total_floorsG);

                                    if (floorIdx == -1) {
                                        if (intendedPos[1] <= oldFloorHeight) {
                                            intendedPos[1] = oldFloorHeight;
                                            landed = true;
                                        }
                                        falling = false;
                                        break;
                                    }
                                    else if (newFloor->normal[1] < 0.7880108) {
                                        falling = false;
                                        break;
                                    }
                                    else if (intendedPos[1] <= floorHeight) {
                                        intendedPos[1] = floorHeight;

                                        if (!newFloor->is_lava) {
                                            landed = true;
                                        }

                                        falling = false;
                                        break;
                                    }
                                    else if (intendedPos[1] < -1357.0) {
                                        falling = false;
                                        break;
                                    }
                                    else if (intendedPos[0] < INT_MIN || intendedPos[0] > INT_MAX || intendedPos[2] < INT_MIN || intendedPos[2] > INT_MAX) {
                                        falling = false;
                                        break;
                                    }
                                }

                                yVel = fminf(yVel - 4.0f, 75.0f);
                            }

                            if (landed && intendedPos[1] >= -1357.0) {
                                int solIdx = atomicAdd(&nBDSolutions, 1);

                                if (solIdx < MAX_BD_SOLUTIONS) {
                                    BDSolution* solution = &(bdSolutions[solIdx]);
                                    solution->slideSolutionIdx = idx;
                                    solution->cameraYaw;
                                    solution->stickX = rawX;
                                    solution->stickY = rawY;
                                    solution->landingPosition[0] = intendedPos[0];
                                    solution->landingPosition[1] = intendedPos[1];
                                    solution->landingPosition[2] = intendedPos[2];
                                    solution->postSlideSpeed = postSlideSpeed;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


// Overall Run Bruteforcer Function

void run_slide_kick_bruteforcer(int g, int h, int i, int j, float normX, float normY, float normZ, short* host_tris, float* host_norms, short* dev_tris, float* dev_norms, SKSolStruct s, float normal_offsets_cpu[4][3], short* floorPoints, short* devFloorPoints, bool* squishEdges, bool* devSquishEdges, std::ofstream &wf, char* normalStages)
{
    int current_normal_index = ((g * nSamplesNY + h) * nSamplesNX + i) * nSamplesNZ + j;

    Vec3f startNormal = { normX, normY, normZ };
    set_platform_normal << <1, 1 >> > (normX, normY, normZ);

    int sumPlatformSolutions = 0;
    int sumUpwarpSolutions = 0;
    int sumSK6SolutionsCPU = 0;
    int sumSKUWSolutionsCPU = 0;
    int sumSpeedSolutionsSKCPU = 0;
    int sum10KSolutionsCPU = 0;
    int sumSlideSolutionsCPU = 0;
    int sumBDSolutionsCPU = 0;

    if (subSolutionPrintingMode == 2)
        printf("---------------------------------------\nTesting Normal: %g, %g, %g\n  Index: %d, %d, %d, %d\n", normX, normY, normZ, g, h, i, j);

    for (int t = 0; t < 4; t++) {
        Platform platform = Platform(platformPos[0], platformPos[1], platformPos[2], startNormal);

        float ceilingNormals[4] = { platform.ceilings[0].normal[1], platform.ceilings[1].normal[1], platform.ceilings[2].normal[1], platform.ceilings[3].normal[1] };
        bool squishTest = (ceilingNormals[0] > -0.5f) || (ceilingNormals[1] > -0.5f) || (ceilingNormals[2] > -0.5f) || (ceilingNormals[3] > -0.5f);

        if (!squishTest) {
            break;
        }

        set_squish_ceilings << <1, 1 >> > (ceilingNormals[0], ceilingNormals[1], ceilingNormals[2], ceilingNormals[3]);
        Vec3f position = { 0.0f, 0.0f, 0.0f };

        platform.platform_logic(position);

        for (int x = 0; x < 2; x++) {
            for (int y = 0; y < 3; y++) {
                host_tris[9 * x + 3 * y] = platform.triangles[x].vectors[y][0];
                host_tris[9 * x + 3 * y + 1] = platform.triangles[x].vectors[y][1];
                host_tris[9 * x + 3 * y + 2] = platform.triangles[x].vectors[y][2];
                host_norms[3 * x + y] = platform.triangles[x].normal[y];
            }
        }

        cudaMemcpy(dev_tris, host_tris, 18 * sizeof(short), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_norms, host_norms, 6 * sizeof(float), cudaMemcpyHostToDevice);

        set_start_triangle << <1, 1 >> > (dev_tris, dev_norms);

        platform.normal[0] += normal_offsets_cpu[t][0];
        platform.normal[1] += normal_offsets_cpu[t][1];
        platform.normal[2] += normal_offsets_cpu[t][2];

        for (int k = 2; k < nPUFrames; k++) {
            platform.platform_logic(position);
        }

        float minX = INT16_MAX;
        float maxX = INT16_MIN;
        float minZ = INT16_MAX;
        float maxZ = INT16_MIN;

        for (int k = 0; k < platform.triangles.size(); k++) {
            minX = fminf(fminf(fminf(minX, platform.triangles[k].vectors[0][0]), platform.triangles[k].vectors[1][0]), platform.triangles[k].vectors[2][0]);
            maxX = fmaxf(fmaxf(fmaxf(maxX, platform.triangles[k].vectors[0][0]), platform.triangles[k].vectors[1][0]), platform.triangles[k].vectors[2][0]);
            minZ = fminf(fminf(fminf(minZ, platform.triangles[k].vectors[0][2]), platform.triangles[k].vectors[1][2]), platform.triangles[k].vectors[2][2]);
            maxZ = fmaxf(fmaxf(fmaxf(maxZ, platform.triangles[k].vectors[0][2]), platform.triangles[k].vectors[1][2]), platform.triangles[k].vectors[2][2]);
        }

        int nX = round((maxX - minX) / deltaX) + 1;
        int nZ = round((maxZ - minZ) / deltaZ) + 1;

        int nPlatSolutionsCPU = 0;
        int nUpwarpSolutionsCPU = 0;

        cudaMemcpyToSymbol(nPlatSolutions, &nPlatSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

        long long int nBlocks = (nX * nZ + nThreads - 1) / nThreads;

        cudaFunc << <nBlocks, nThreads >> > (minX, deltaX, minZ, deltaZ, nX, nZ, platform.normal[0], platform.normal[1], platform.normal[2], maxFrames);

        cudaMemcpyFromSymbol(&nPlatSolutionsCPU, nPlatSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);

        if (nPlatSolutionsCPU > 0) {
            if (nPlatSolutionsCPU > MAX_PLAT_SOLUTIONS) {
                fprintf(stderr, "Warning: Number of platform solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                nPlatSolutionsCPU = MAX_PLAT_SOLUTIONS;
            }

            normalStages[current_normal_index] = max(1,normalStages[current_normal_index]);
            sumPlatformSolutions += nPlatSolutionsCPU;

            nBlocks = (nPlatSolutionsCPU + nThreads - 1) / nThreads;

            cudaMemcpyToSymbol(nUpwarpSolutions, &nUpwarpSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

            find_upwarp_solutions << <nBlocks, nThreads >> > (1000000000.0f);

            cudaMemcpyFromSymbol(&nUpwarpSolutionsCPU, nUpwarpSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
        }

        if (nUpwarpSolutionsCPU > 0) {
            if (nUpwarpSolutionsCPU > MAX_UPWARP_SOLUTIONS) {
                fprintf(stderr, "Warning: Number of upwarp solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                nUpwarpSolutionsCPU = MAX_UPWARP_SOLUTIONS;
            }

            normalStages[current_normal_index] = max(2, normalStages[current_normal_index]);
            sumUpwarpSolutions += nUpwarpSolutionsCPU;

            bool sameNormal = host_norms[1] == host_norms[4];

            for (int x = 0; x < (sameNormal ? 1 : 2); x++) {
                for (int y = 0; y < 3; y++) {
                    floorPoints[3 * y] = host_tris[9 * x + 3 * y];
                    floorPoints[3 * y + 1] = host_tris[9 * x + 3 * y + 1];
                    floorPoints[3 * y + 2] = host_tris[9 * x + 3 * y + 2];
                }

                if (sameNormal) {
                    floorPoints[9] = host_tris[15];
                    floorPoints[10] = host_tris[16];
                    floorPoints[11] = host_tris[17];
                }

                squishEdges[0] = (i == 0) ? ceilingNormals[2] > -0.5f : false;
                squishEdges[1] = ceilingNormals[(i == 0) ? 0 : 1] > -0.5f;
                squishEdges[2] = (sameNormal || i == 1) ? ceilingNormals[sameNormal ? 1 : 3] > -0.5f : false;
                squishEdges[3] = sameNormal ? ceilingNormals[3] > -0.5f : false;

                cudaMemcpy(devSquishEdges, squishEdges, 4 * sizeof(bool), cudaMemcpyHostToDevice);

                int nSK6SolutionsCPU = 0;
                int nSKUWSolutionsCPU = 0;
                int nSpeedSolutionsSKCPU = 0;
                int n10KSolutionsCPU = 0;
                int nSlideSolutionsCPU = 0;
                int nBDSolutionsCPU = 0;

                cudaMemcpyToSymbol(nSK6Solutions, &nSK6SolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);
                cudaMemcpyToSymbol(nSKUWSolutions, &nSKUWSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);
                cudaMemcpyToSymbol(nSpeedSolutionsSK, &nSpeedSolutionsSKCPU, sizeof(int), 0, cudaMemcpyHostToDevice);
                cudaMemcpyToSymbol(n10KSolutionsSK, &n10KSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);
                cudaMemcpyToSymbol(nSlideSolutions, &nSlideSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);
                cudaMemcpyToSymbol(nBDSolutions, &nBDSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

                find_slide_kick_setup_triangle(floorPoints, devFloorPoints, sameNormal ? 4 : 3, host_norms[3 * x + 1], t, maxSpeedSK, nThreads);

                cudaMemcpyFromSymbol(&nSK6SolutionsCPU, nSK6Solutions, sizeof(int), 0, cudaMemcpyDeviceToHost);

                if (nSK6SolutionsCPU > 0) {
                    if (nSK6SolutionsCPU > MAX_SK_PHASE_SIX) {
                        fprintf(stderr, "Warning: Number of phase 6 solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                        nSK6SolutionsCPU = MAX_SK_PHASE_SIX;
                    }

                    normalStages[current_normal_index] = max(3, normalStages[current_normal_index]);
                    sumSK6SolutionsCPU += nSK6SolutionsCPU;

                    nBlocks = (nUpwarpSolutionsCPU + nThreads - 1) / nThreads;

                    cudaMemcpyToSymbol(nSKUWSolutions, &nSKUWSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

                    find_sk_upwarp_solutions << <nBlocks, nThreads >> > ();

                    cudaMemcpyFromSymbol(&nSKUWSolutionsCPU, nSKUWSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
                }

                if (nSKUWSolutionsCPU > 0) {
                    if (nSKUWSolutionsCPU > MAX_SK_UPWARP_SOLUTIONS) {
                        fprintf(stderr, "Warning: Number of slide kick upwarp solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                        nSKUWSolutionsCPU = MAX_SK_UPWARP_SOLUTIONS;
                    }

                    normalStages[current_normal_index] = max(4, normalStages[current_normal_index]);
                    sumSKUWSolutionsCPU += nSKUWSolutionsCPU;

                    nBlocks = (nSKUWSolutionsCPU + nThreads - 1) / nThreads;

                    cudaMemcpyToSymbol(nSpeedSolutionsSK, &nSpeedSolutionsSKCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

                    find_speed_solutions << <nBlocks, nThreads >> > ();

                    cudaMemcpyFromSymbol(&nSpeedSolutionsSKCPU, nSpeedSolutionsSK, sizeof(int), 0, cudaMemcpyDeviceToHost);
                }

                if (nSpeedSolutionsSKCPU > 0) {
                    if (nSpeedSolutionsSKCPU > MAX_SPEED_SOLUTIONS_SK) {
                        fprintf(stderr, "Warning: Number of speed solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                        nSpeedSolutionsSKCPU = MAX_SPEED_SOLUTIONS_SK;
                    }

                    normalStages[current_normal_index] = max(5, normalStages[current_normal_index]);
                    sumSpeedSolutionsSKCPU += nSpeedSolutionsSKCPU;

                    nBlocks = (nSpeedSolutionsSKCPU + nThreads - 1) / nThreads;

                    cudaMemcpyToSymbol(n10KSolutionsSK, &n10KSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

                    test_speed_solution << <nBlocks, nThreads >> > (devFloorPoints, devSquishEdges, sameNormal ? 4 : 3, host_norms[3 * x + 1]);

                    cudaMemcpyFromSymbol(&n10KSolutionsCPU, n10KSolutionsSK, sizeof(int), 0, cudaMemcpyDeviceToHost);
                }

                if (n10KSolutionsCPU > 0) {
                    if (n10KSolutionsCPU > MAX_10K_SOLUTIONS_SK) {
                        fprintf(stderr, "Warning: Number of 10K solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                        n10KSolutionsCPU = MAX_10K_SOLUTIONS_SK;
                    }
                    normalStages[current_normal_index] = max(6, normalStages[current_normal_index]);
                    sum10KSolutionsCPU += n10KSolutionsCPU;

                    nBlocks = (n10KSolutionsCPU + nThreads - 1) / nThreads;

                    cudaMemcpyToSymbol(nSlideSolutions, &nSlideSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

                    find_slide_solutions << <nBlocks, nThreads >> > ();

                    cudaMemcpyFromSymbol(&nSlideSolutionsCPU, nSlideSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
                }

                if (nSlideSolutionsCPU > 0) {
                    if (nSlideSolutionsCPU > MAX_SLIDE_SOLUTIONS) {
                        fprintf(stderr, "Warning: Number of slide solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                        nSlideSolutionsCPU = MAX_SLIDE_SOLUTIONS;
                    }

                    normalStages[current_normal_index] = max(7, normalStages[current_normal_index]);
                    sumSlideSolutionsCPU += nSlideSolutionsCPU;

                    nBlocks = (nSlideSolutionsCPU + nThreads - 1) / nThreads;

                    cudaMemcpyToSymbol(nBDSolutions, &nBDSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

                    find_breakdance_solutions << <nBlocks, nThreads >> > ();

                    cudaMemcpyFromSymbol(&nBDSolutionsCPU, nBDSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
                }

                if (nBDSolutionsCPU > 0) {
                    if (nBDSolutionsCPU > MAX_BD_SOLUTIONS) {
                        fprintf(stderr, "Warning: Number of breakdance solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                        nBDSolutionsCPU = MAX_BD_SOLUTIONS;
                    }

                    normalStages[current_normal_index] = max(8, normalStages[current_normal_index]);
                    sumBDSolutionsCPU += nBDSolutionsCPU;

                    int nSK1SolutionsCPU = 0;
                    int nSK2ASolutionsCPU = 0;
                    int nSK2BSolutionsCPU = 0;
                    int nSK2CSolutionsCPU = 0;
                    int nSK2DSolutionsCPU = 0;
                    int nSK3SolutionsCPU = 0;
                    int nSK4SolutionsCPU = 0;
                    int nSK5SolutionsCPU = 0;
                    int nSK6SolutionsCPU = 0;

                    cudaMemcpyFromSymbol(&nSK1SolutionsCPU, nSK1Solutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
                    cudaMemcpyFromSymbol(&nSK2ASolutionsCPU, nSK2ASolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
                    cudaMemcpyFromSymbol(&nSK2BSolutionsCPU, nSK2BSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
                    cudaMemcpyFromSymbol(&nSK2CSolutionsCPU, nSK2CSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
                    cudaMemcpyFromSymbol(&nSK2DSolutionsCPU, nSK2DSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
                    cudaMemcpyFromSymbol(&nSK3SolutionsCPU, nSK3Solutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
                    cudaMemcpyFromSymbol(&nSK4SolutionsCPU, nSK4Solutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
                    cudaMemcpyFromSymbol(&nSK5SolutionsCPU, nSK5Solutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
                    cudaMemcpyFromSymbol(&nSK6SolutionsCPU, nSK6Solutions, sizeof(int), 0, cudaMemcpyDeviceToHost);

                    nSK1SolutionsCPU = min(nSK1SolutionsCPU, MAX_SK_PHASE_ONE);
                    nSK2ASolutionsCPU = min(nSK2ASolutionsCPU, MAX_SK_PHASE_TWO_A);
                    nSK2BSolutionsCPU = min(nSK2BSolutionsCPU, MAX_SK_PHASE_TWO_B);
                    nSK2CSolutionsCPU = min(nSK2CSolutionsCPU, MAX_SK_PHASE_TWO_C);
                    nSK2DSolutionsCPU = min(nSK2DSolutionsCPU, MAX_SK_PHASE_TWO_D);
                    nSK3SolutionsCPU = min(nSK3SolutionsCPU, MAX_SK_PHASE_THREE);
                    nSK4SolutionsCPU = min(nSK4SolutionsCPU, MAX_SK_PHASE_FOUR);
                    nSK5SolutionsCPU = min(nSK5SolutionsCPU, MAX_SK_PHASE_FIVE);
                    nSK6SolutionsCPU = min(nSK6SolutionsCPU, MAX_SK_PHASE_SIX);

                    struct BDSolution* bdSolutionsCPU = (struct BDSolution*)std::malloc(nBDSolutionsCPU * sizeof(struct BDSolution));
                    struct SlideSolution* slideSolutionsCPU = (struct SlideSolution*)std::malloc(nSlideSolutionsCPU * sizeof(struct SlideSolution));
                    struct TenKSolutionSK* tenKSolutionsCPU = (struct TenKSolutionSK*)std::malloc(n10KSolutionsCPU * sizeof(struct TenKSolutionSK));
                    struct SpeedSolutionSK* speedSolutionsCPU = (struct SpeedSolutionSK*)std::malloc(nSpeedSolutionsSKCPU * sizeof(struct SpeedSolutionSK));
                    struct SKUpwarpSolution* skuwSolutionsCPU = (struct SKUpwarpSolution*)std::malloc(nSKUWSolutionsCPU * sizeof(struct SKUpwarpSolution));

                    struct PlatformSolution* platSolutionsCPU = (struct PlatformSolution*)std::malloc(nPlatSolutionsCPU * sizeof(struct PlatformSolution));
                    struct UpwarpSolution* upwarpSolutionsCPU = (struct UpwarpSolution*)std::malloc(nUpwarpSolutionsCPU * sizeof(struct UpwarpSolution));

                    struct SKPhase1* sk1SolutionsCPU = (struct SKPhase1*)std::malloc(nSK1SolutionsCPU * sizeof(struct SKPhase1));
                    struct SKPhase2* sk2ASolutionsCPU = (struct SKPhase2*)std::malloc(nSK2ASolutionsCPU * sizeof(struct SKPhase2));
                    struct SKPhase2* sk2BSolutionsCPU = (struct SKPhase2*)std::malloc(nSK2BSolutionsCPU * sizeof(struct SKPhase2));
                    struct SKPhase2* sk2CSolutionsCPU = (struct SKPhase2*)std::malloc(nSK2CSolutionsCPU * sizeof(struct SKPhase2));
                    struct SKPhase2* sk2DSolutionsCPU = (struct SKPhase2*)std::malloc(nSK2DSolutionsCPU * sizeof(struct SKPhase2));
                    struct SKPhase3* sk3SolutionsCPU = (struct SKPhase3*)std::malloc(nSK3SolutionsCPU * sizeof(struct SKPhase3));
                    struct SKPhase4* sk4SolutionsCPU = (struct SKPhase4*)std::malloc(nSK4SolutionsCPU * sizeof(struct SKPhase4));
                    struct SKPhase5* sk5SolutionsCPU = (struct SKPhase5*)std::malloc(nSK5SolutionsCPU * sizeof(struct SKPhase5));
                    struct SKPhase6* sk6SolutionsCPU = (struct SKPhase6*)std::malloc(nSK6SolutionsCPU * sizeof(struct SKPhase6));

                    cudaMemcpy(bdSolutionsCPU, s.bdSolutions, nBDSolutionsCPU * sizeof(struct BDSolution), cudaMemcpyDeviceToHost);
                    cudaMemcpy(slideSolutionsCPU, s.slideSolutions, nSlideSolutionsCPU * sizeof(struct SlideSolution), cudaMemcpyDeviceToHost);
                    cudaMemcpy(tenKSolutionsCPU, s.tenKSolutions, n10KSolutionsCPU * sizeof(struct TenKSolutionSK), cudaMemcpyDeviceToHost);
                    cudaMemcpy(speedSolutionsCPU, s.speedSolutions, nSpeedSolutionsSKCPU * sizeof(struct SpeedSolutionSK), cudaMemcpyDeviceToHost);
                    cudaMemcpy(skuwSolutionsCPU, s.skuwSolutions, nSKUWSolutionsCPU * sizeof(struct SKUpwarpSolution), cudaMemcpyDeviceToHost);

                    cudaMemcpy(upwarpSolutionsCPU, s.upwarpSolutions, nUpwarpSolutionsCPU * sizeof(struct UpwarpSolution), cudaMemcpyDeviceToHost);
                    cudaMemcpy(platSolutionsCPU, s.platSolutions, nPlatSolutionsCPU * sizeof(struct PlatformSolution), cudaMemcpyDeviceToHost);

                    cudaMemcpy(sk1SolutionsCPU, s.sk1Solutions, nSK1SolutionsCPU * sizeof(struct SKPhase1), cudaMemcpyDeviceToHost);
                    cudaMemcpy(sk2ASolutionsCPU, s.sk2ASolutions, nSK2ASolutionsCPU * sizeof(struct SKPhase2), cudaMemcpyDeviceToHost);
                    cudaMemcpy(sk2BSolutionsCPU, s.sk2BSolutions, nSK2BSolutionsCPU * sizeof(struct SKPhase2), cudaMemcpyDeviceToHost);
                    cudaMemcpy(sk2CSolutionsCPU, s.sk2CSolutions, nSK2CSolutionsCPU * sizeof(struct SKPhase2), cudaMemcpyDeviceToHost);
                    cudaMemcpy(sk2DSolutionsCPU, s.sk2DSolutions, nSK2DSolutionsCPU * sizeof(struct SKPhase2), cudaMemcpyDeviceToHost);
                    cudaMemcpy(sk3SolutionsCPU, s.sk3Solutions, nSK3SolutionsCPU * sizeof(struct SKPhase3), cudaMemcpyDeviceToHost);
                    cudaMemcpy(sk4SolutionsCPU, s.sk4Solutions, nSK4SolutionsCPU * sizeof(struct SKPhase4), cudaMemcpyDeviceToHost);
                    cudaMemcpy(sk5SolutionsCPU, s.sk5Solutions, nSK5SolutionsCPU * sizeof(struct SKPhase5), cudaMemcpyDeviceToHost);
                    cudaMemcpy(sk6SolutionsCPU, s.sk6Solutions, nSK6SolutionsCPU * sizeof(struct SKPhase6), cudaMemcpyDeviceToHost);

                    for (int l = 0; l < nBDSolutionsCPU; l++) {
                        struct BDSolution* bdSol = &(bdSolutionsCPU[l]);
                        struct SlideSolution* slideSol = &(slideSolutionsCPU[bdSol->slideSolutionIdx]);
                        struct TenKSolutionSK* tenKSol = &(tenKSolutionsCPU[slideSol->tenKSolutionIdx]);
                        struct SpeedSolutionSK* speedSol = &(speedSolutionsCPU[tenKSol->speedSolutionIdx]);
                        struct SKUpwarpSolution* skuwSol = &(skuwSolutionsCPU[speedSol->skuwSolutionIdx]);
                        struct UpwarpSolution* uwSol = &(upwarpSolutionsCPU[skuwSol->uwIdx]);
                        struct PlatformSolution* platSol = &(platSolutionsCPU[uwSol->platformSolutionIdx]);
                        struct SKPhase6* p6Sol = &(sk6SolutionsCPU[skuwSol->skIdx]);
                        struct SKPhase5* p5Sol = &(sk5SolutionsCPU[p6Sol->p5Idx]);
                        struct SKPhase4* p4Sol = &(sk4SolutionsCPU[p5Sol->p4Idx]);
                        struct SKPhase3* p3Sol = &(sk3SolutionsCPU[p4Sol->p3Idx]);
                        struct SKPhase2* p2Sol = (p3Sol->p2Type / 2 == 0) ? ((p3Sol->p2Type % 2 == 0) ? &(sk2ASolutionsCPU[p3Sol->p2Idx]) : &(sk2BSolutionsCPU[p3Sol->p2Idx])) : ((p3Sol->p2Type % 2 == 0) ? &(sk2CSolutionsCPU[p3Sol->p2Idx]) : &(sk2DSolutionsCPU[p3Sol->p2Idx]));
                        struct SKPhase1* p1Sol = &(sk1SolutionsCPU[p2Sol->p1Idx]);

                        printf("---------------------------------------\nFound Solution:\n---------------------------------------\n    Start Position: %.10g, %.10g, %.10g\n    Frame 1 Position: %.10g, %.10g, %.10g\n    Frame 2 Position: %.10g, %.10g, %.10g\n    Return Position: %.10g, %.10g, %.10g\n    PU Departure Speed: %.10g (x=%.10g, z=%.10g)\n    PU Return Speed: %.10g (x=%.10g, z=%.10g)\n    Frame 1 Q-steps: %d\n    Frame 2 Q-steps: %d\n    Frame 3 Q-steps: %d\n", tenKSol->startPosition[0], tenKSol->startPosition[1], tenKSol->startPosition[2], tenKSol->frame1Position[0], tenKSol->frame1Position[1], tenKSol->frame1Position[2], tenKSol->frame2Position[0], tenKSol->frame2Position[1], tenKSol->frame2Position[2], platSol->returnPosition[0], platSol->returnPosition[1], platSol->returnPosition[2], tenKSol->pre10KSpeed, tenKSol->pre10KVel[0], tenKSol->pre10KVel[1], speedSol->returnSpeed, tenKSol->returnVel[0], tenKSol->returnVel[1], 4, p1Sol->q2, 1);
                        printf("    10k Stick X: %d\n    10k Stick Y: %d\n    Frame 2 HAU: %d\n    10k Camera Yaw: %d\n    Start Floor Normal: %.10g, %.10g, %.10g\n", ((p5Sol->stickX == 0) ? 0 : ((p5Sol->stickX < 0) ? p5Sol->stickX - 6 : p5Sol->stickX + 6)), ((p5Sol->stickY == 0) ? 0 : ((p5Sol->stickY < 0) ? p5Sol->stickY - 6 : p5Sol->stickY + 6)), p2Sol->f2Angle, p4Sol->cameraYaw, host_norms[3 * x], host_norms[3 * x + 1], host_norms[3 * x + 2]);
                        printf("---------------------------------------\n    Tilt Frames: %d\n    Post-Tilt Platform Normal: %.10g, %.10g, %.10g\n    Post-Tilt Position: %.10g, %.10g, %.10g\n    Pre-Upwarp Position: %.10g, %.10g, %.10g\n    Post-Upwarp Position: %.10g, %.10g, %.10g\n    Upwarp PU X: %d\n    Upwarp PU Z: %d\n    Upwarp Slide Facing Angle: %d\n    Upwarp Slide Intended Mag: %.10g\n    Upwarp Slide Intended DYaw: %d\n", platSol->nFrames, platSol->endNormal[0], platSol->endNormal[1], platSol->endNormal[2], platSol->endPosition[0], platSol->endPosition[1], platSol->endPosition[2], slideSol->preUpwarpPosition[0], slideSol->preUpwarpPosition[1], slideSol->preUpwarpPosition[2], slideSol->upwarpPosition[0], slideSol->upwarpPosition[1], slideSol->upwarpPosition[2], uwSol->pux, uwSol->puz, slideSol->angle, slideSol->stickMag, slideSol->intendedDYaw);
                        printf("---------------------------------------\n    Post-Breakdance Camera Yaw: %d\n    Post-Breakdance Stick X: %d\n    Post-Breakdance Stick Y: %d\n    Landing Position: %.10g, %.10g, %.10g\n    Landing Speed: %.10g\n---------------------------------------\n\n\n", bdSol->cameraYaw, bdSol->stickX, bdSol->stickY, bdSol->landingPosition[0], bdSol->landingPosition[1], bdSol->landingPosition[2], bdSol->postSlideSpeed);

                        wf << normX << ", " << normY << ", " << normZ << ", ";
                        wf << tenKSol->startPosition[0] << ", " << tenKSol->startPosition[1] << ", " << tenKSol->startPosition[2] << ", ";
                        wf << tenKSol->frame1Position[0] << ", " << tenKSol->frame1Position[1] << ", " << tenKSol->frame1Position[2] << ", ";
                        wf << tenKSol->frame2Position[0] << ", " << tenKSol->frame2Position[1] << ", " << tenKSol->frame2Position[2] << ", ";
                        wf << platSol->returnPosition[0] << ", " << platSol->returnPosition[1] << ", " << platSol->returnPosition[2] << ", ";
                        wf << tenKSol->pre10KSpeed << ", " << tenKSol->pre10KVel[0] << ", " << tenKSol->pre10KVel[1] << ", ";
                        wf << speedSol->returnSpeed << ", " << tenKSol->returnVel[0] << ", " << tenKSol->returnVel[1] << ", ";
                        wf << 4 << ", " << p1Sol->q2 << ", " << 1 << ", ";
                        wf << ((p5Sol->stickX == 0) ? 0 : ((p5Sol->stickX < 0) ? p5Sol->stickX - 6 : p5Sol->stickX + 6)) << ", " << ((p5Sol->stickY == 0) ? 0 : ((p5Sol->stickY < 0) ? p5Sol->stickY - 6 : p5Sol->stickY + 6)) << ", ";
                        wf << p2Sol->f2Angle << ", " << p4Sol->cameraYaw << ", ";
                        wf << host_norms[3 * x] << ", " << host_norms[3 * x + 1] << ", " << host_norms[3 * x + 2] << ", ";
                        wf << platSol->nFrames << ", ";
                        wf << platSol->endNormal[0] << ", " << platSol->endNormal[1] << ", " << platSol->endNormal[2] << ", ";
                        wf << platSol->endPosition[0] << ", " << platSol->endPosition[1] << ", " << platSol->endPosition[2] << ", ";
                        wf << slideSol->preUpwarpPosition[0] << ", " << slideSol->preUpwarpPosition[1] << ", " << slideSol->preUpwarpPosition[2] << ", ";
                        wf << slideSol->upwarpPosition[0] << ", " << slideSol->upwarpPosition[1] << ", " << slideSol->upwarpPosition[2] << ", ";
                        wf << uwSol->pux << ", " << uwSol->puz << ", ";
                        wf << slideSol->angle << ", " << slideSol->stickMag << ", " << slideSol->intendedDYaw << ", ";
                        wf << bdSol->cameraYaw << ", ";
                        wf << bdSol->stickX << ", " << bdSol->stickY << ", ";
                        wf << bdSol->landingPosition[0] << ", " << bdSol->landingPosition[1] << ", " << bdSol->landingPosition[2] << ", ";
                        wf << bdSol->postSlideSpeed << std::endl;
                    }

                    free(bdSolutionsCPU);
                    free(slideSolutionsCPU);
                    free(tenKSolutionsCPU);
                    free(speedSolutionsCPU);
                    free(skuwSolutionsCPU);
                    free(upwarpSolutionsCPU);
                    free(platSolutionsCPU);
                    free(sk1SolutionsCPU);
                    free(sk2ASolutionsCPU);
                    free(sk2BSolutionsCPU);
                    free(sk2CSolutionsCPU);
                    free(sk2DSolutionsCPU);
                    free(sk3SolutionsCPU);
                    free(sk4SolutionsCPU);
                    free(sk5SolutionsCPU);
                    free(sk6SolutionsCPU);
                }
            }
        }
    }

    if (subSolutionPrintingMode == 1)
    {
        if (sum10KSolutionsCPU > 0)
            printf("# 10K Solutions (%d, %d, %d, %d): %d\n\n", g, h, i, j, sum10KSolutionsCPU);

        if (sumSlideSolutionsCPU > 0)
            printf("# Slide Solutions (%d, %d, %d, %d): %d\n\n", g, h, i, j, sumSlideSolutionsCPU);

        if (sumBDSolutionsCPU > 0)
            printf("# Slide Solutions (%d, %d, %d, %d): %d\n\n", g, h, i, j, sumSlideSolutionsCPU);
    }

    if (subSolutionPrintingMode == 2)
    {
        printf("  Stage 1 Solutions: %d\n", sumPlatformSolutions);
        printf("  Stage 2 Solutions: %d\n", sumUpwarpSolutions);
        printf("  Stage 3 Solutions: %d\n", sumSK6SolutionsCPU);
        printf("  Stage 4 Solutions: %d\n", sumSKUWSolutionsCPU);
        printf("  Stage 5 Solutions: %d\n", sumSpeedSolutionsSKCPU);
        printf("  Stage 6 Solutions: %d\n", sum10KSolutionsCPU);
        printf("  Stage 7 Solutions: %d\n", sumSlideSolutionsCPU);
        printf("  Stage 8 Solutions: %d\n", sumBDSolutionsCPU);
    }
}