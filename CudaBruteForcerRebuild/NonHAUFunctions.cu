#include "NonHAUFunctions.cuh"

#include "math.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#include "Platform.cuh"
#include "vmath.hpp"

#include "BruteforceVariables.cuh"
#include "CommonFunctions.cuh"
#include "Floors.cuh"
#include "PlatformSolutionFunctions.cuh"
#include "UpwarpSolutionFunctions.cuh"
#include "RunParameters.hpp"


__global__ void copy_solution_pointers_non_hau(NonHAUSolStruct s) {
    platSolutions = s.platSolutions;
    upwarpSolutions = s.upwarpSolutions;
    puSolutions = s.puSolutions;
    tenKSolutionsNonHAU = s.tenKSolutions;
}

void init_solution_structs_non_hau(NonHAUSolStruct* s) {
    cudaMalloc((void**)&s->platSolutions, MAX_PLAT_SOLUTIONS * sizeof(PlatformSolution));
    cudaMalloc((void**)&s->upwarpSolutions, MAX_UPWARP_SOLUTIONS * sizeof(UpwarpSolution));
    cudaMalloc((void**)&s->puSolutions, MAX_PU_SOLUTIONS * sizeof(PUSolution));
    cudaMalloc((void**)&s->tenKSolutions, MAX_10K_SOLUTIONS_NON_HAU * sizeof(TenKSolutionNonHAU));

    copy_solution_pointers_non_hau << <1, 1 >> >(*s);
}

void free_solution_pointers_non_hau(NonHAUSolStruct* s) {
    cudaFree(s->platSolutions);
    cudaFree(s->upwarpSolutions);
    cudaFree(s->puSolutions);
    cudaFree(s->tenKSolutions);
}

__device__ bool test_stick_position(int solIdx, int x, int y, float endSpeed, float vel1, float xVel1, float zVel1, int angle, int cameraYaw, float* startPosition, float* oneUpPlatformPosition, float oneUpPlatformXMin, float oneUpPlatformXMax, float oneUpPlatformYMin, float oneUpPlatformYMax, float oneUpPlatformZMin, float oneUpPlatformZMax, float oneUpPlatformNormalX, float oneUpPlatformNormalY, int f, float* frame1Position, float* returnPosition, int q1, int q2, int q3) {
    bool foundSolution = false;

    float testStartPosition[3] = { startPosition[0], startPosition[1], startPosition[2] };
    float testFrame1Position[3] = { frame1Position[0], frame1Position[1], frame1Position[2] };
    float testOneUpPlatformPosition[3] = { oneUpPlatformPosition[0], oneUpPlatformPosition[1], oneUpPlatformPosition[2] };

    int trueX = (x == 0) ? 0 : ((x < 0) ? x - 6 : x + 6);
    int trueY = (y == 0) ? 0 : ((y < 0) ? y - 6 : y + 6);

    float mag = sqrtf(x * x + y * y);

    float xS = x;
    float yS = y;

    if (mag > 64.0f) {
        xS = xS * (64.0f / mag);
        yS = yS * (64.0f / mag);
        mag = 64.0f;
    }

    float intendedMag = ((mag / 64.0f) * (mag / 64.0f)) * 32.0f;
    int intendedYaw = atan2sG(-yS, xS) + cameraYaw;
    int intendedDYaw = intendedYaw - angle;
    intendedDYaw = (65536 + (intendedDYaw % 65536)) % 65536;

    float xVel2a = xVel1;
    float zVel2a = zVel1;

    float oldSpeed = sqrtf(xVel2a * xVel2a + zVel2a * zVel2a);

    xVel2a += zVel2a * (intendedMag / 32.0f) * gSineTableG[intendedDYaw >> 4] * 0.05f;
    zVel2a -= xVel2a * (intendedMag / 32.0f) * gSineTableG[intendedDYaw >> 4] * 0.05f;

    float newSpeed = sqrtf(xVel2a * xVel2a + zVel2a * zVel2a);

    xVel2a = xVel2a * oldSpeed / newSpeed;
    zVel2a = zVel2a * oldSpeed / newSpeed;

    xVel2a += 7.0f * oneUpPlatformNormalX;

    float forward = gCosineTableG[intendedDYaw >> 4];
    forward *= 0.5f + 0.5f * vel1 / 100.0f;
    float lossFactor = intendedMag / 32.0f * forward * 0.02f + 0.92f;

    xVel2a *= lossFactor;
    zVel2a *= lossFactor;

    float vel2a = -sqrtf(xVel2a * xVel2a + zVel2a * zVel2a);

    bool speedTest = true;

    if (vel2a != endSpeed) {
        double w = intendedMag * gCosineTableG[intendedDYaw >> 4];
        double eqB = (50.0 + 147200.0 / w);
        double eqC = -(320000.0 / w) * endSpeed;
        double eqDet = eqB * eqB - eqC;

        float vel1a;

        if (eqDet > 0) {
            vel1a = sqrt(eqDet) - eqB;

            if (vel1a >= 0) {
                bool searchLoop = true;

                while (searchLoop) {
                    xVel2a = vel1a * gSineTableG[angle >> 4];
                    zVel2a = vel1a * gCosineTableG[angle >> 4];

                    oldSpeed = sqrtf(xVel2a * xVel2a + zVel2a * zVel2a);

                    xVel2a += zVel2a * (intendedMag / 32.0f) * gSineTableG[intendedDYaw >> 4] * 0.05f;
                    zVel2a -= xVel2a * (intendedMag / 32.0f) * gSineTableG[intendedDYaw >> 4] * 0.05f;

                    newSpeed = sqrtf(xVel2a * xVel2a + zVel2a * zVel2a);

                    xVel2a = xVel2a * oldSpeed / newSpeed;
                    zVel2a = zVel2a * oldSpeed / newSpeed;

                    xVel2a += 7.0f * oneUpPlatformNormalX;

                    forward = gCosineTableG[intendedDYaw >> 4] * (0.5f + 0.5f * vel1a / 100.0f);
                    lossFactor = intendedMag / 32.0f * forward * 0.02f + 0.92f;

                    xVel2a *= lossFactor;
                    zVel2a *= lossFactor;

                    float vel2b = -sqrtf(xVel2a * xVel2a + zVel2a * zVel2a);

                    if (vel2b > endSpeed) {
                        vel1a = fmaxf(vel1a + 0.25f, nextafterf(vel1a, INFINITY));
                    }
                    else {
                        if (vel2b == endSpeed) {
                            vel2a = vel2b;
                            vel1 = vel1a;
                            xVel1 = vel1 * gSineTableG[angle >> 4];
                            zVel1 = vel1 * gCosineTableG[angle >> 4];
                            searchLoop = false;
                        }
                        else {
                            speedTest = false;
                            break;
                        }
                    }
                }

                if (speedTest) {
                    testFrame1Position[0] = testOneUpPlatformPosition[0];
                    testFrame1Position[2] = testOneUpPlatformPosition[2];
                    bool inBoundsTest = true;

                    for (int q = 0; q < q2; q++) {
                        testFrame1Position[0] = testFrame1Position[0] - (xVel1 / 4.0f);
                        testFrame1Position[2] = testFrame1Position[2] - (zVel1 / 4.0f);

                        if (!check_inbounds(testFrame1Position)) {
                            inBoundsTest = false;
                            break;
                        }
                    }

                    if (inBoundsTest) {
                        double targetDist = (double)vel1 * (double)startNormals[f][1] / 4.0;
                        double newDist = sqrt((testFrame1Position[0] - testStartPosition[0]) * (testFrame1Position[0] - testStartPosition[0]) + (testFrame1Position[2] - testStartPosition[2]) * (testFrame1Position[2] - testStartPosition[2]));

                        if (fabs(vel1 * startNormals[f][1] / 4.0 - newDist) > 1000.0) {
                            speedTest = false;
                        }

                        testStartPosition[0] = testFrame1Position[0] + (targetDist / newDist) * (testStartPosition[0] - testFrame1Position[0]);
                        testStartPosition[2] = testFrame1Position[2] + (targetDist / newDist) * (testStartPosition[2] - testFrame1Position[2]);

                        if (!check_inbounds(testStartPosition)) {
                            speedTest = false;
                        }

                        int angle2 = atan2sG(testFrame1Position[2] - testStartPosition[2], testFrame1Position[0] - testStartPosition[0]);
                        angle2 = (65536 + angle2) % 65536;

                        if (angle != angle2) {
                            speedTest = false;
                        }
                    }
                    else {
                        speedTest = false;
                    }
                }
            }
            else {
                speedTest = false;
            }
        }
        else {
            speedTest = false;
        }
    }

    float predictedReturnPosition[3] = { testOneUpPlatformPosition[0] + (oneUpPlatformNormalY * xVel2a / 4.0f), testOneUpPlatformPosition[1], testOneUpPlatformPosition[2] + (oneUpPlatformNormalY * zVel2a / 4.0f) };

    for (int q = 1; q < q3; q++) {
        predictedReturnPosition[0] = predictedReturnPosition[0] + (xVel2a / 4.0f);
        predictedReturnPosition[2] = predictedReturnPosition[2] + (xVel2a / 4.0f);
    }

    if (speedTest && fabs(predictedReturnPosition[0] - returnPosition[0]) < 1000 && fabs(predictedReturnPosition[2] - returnPosition[2]) < 1000) {
        float xShift = predictedReturnPosition[0] - returnPosition[0];
        float zShift = predictedReturnPosition[2] - returnPosition[2];

        testOneUpPlatformPosition[0] = testOneUpPlatformPosition[0] - xShift;
        testOneUpPlatformPosition[1] = oneUpPlatformYMin + (oneUpPlatformYMax - oneUpPlatformYMin) * ((short)(int)testOneUpPlatformPosition[0] - oneUpPlatformXMin) / (oneUpPlatformXMax - oneUpPlatformXMin);
        testOneUpPlatformPosition[2] = testOneUpPlatformPosition[2] - zShift;

        if ((int)(short)testOneUpPlatformPosition[0] >= oneUpPlatformXMin && (int)(short)testOneUpPlatformPosition[0] <= oneUpPlatformXMax && (int)(short)testOneUpPlatformPosition[2] >= oneUpPlatformZMin && (int)(short)testOneUpPlatformPosition[2] <= oneUpPlatformZMax) {
            bool floorTest = true;

            predictedReturnPosition[0] = testOneUpPlatformPosition[0] + (oneUpPlatformNormalY * xVel2a / 4.0f);
            predictedReturnPosition[1] = testOneUpPlatformPosition[1];
            predictedReturnPosition[2] = testOneUpPlatformPosition[2] + (oneUpPlatformNormalY * zVel2a / 4.0f);

            if (q3 > 1) {
                for (int q = 1; q < q3; q++) {
                    float floorHeight;
                    float floorNormalY;

                    if (!fast_floor_check(predictedReturnPosition, floorHeight, floorNormalY)) {
                        floorTest = false;
                        break;
                    }

                    predictedReturnPosition[0] = predictedReturnPosition[0] + floorNormalY * (xVel2a / 4.0f);
                    predictedReturnPosition[1] = floorHeight;
                    predictedReturnPosition[2] = predictedReturnPosition[2] + floorNormalY * (xVel2a / 4.0f);
                }
            }

            if (floorTest && predictedReturnPosition[0] == returnPosition[0] && predictedReturnPosition[2] == returnPosition[2] && predictedReturnPosition[1] < returnPosition[1] && predictedReturnPosition[1] > fmaxf(-2971.0f, returnPosition[1] - 78.0f)) {
                testStartPosition[0] = testStartPosition[0] - xShift;
                testStartPosition[2] = testStartPosition[2] - zShift;

                testFrame1Position[0] = testFrame1Position[0] - xShift;
                testFrame1Position[2] = testFrame1Position[2] - zShift;

                float intersectionPoints[2][3];
                int intersections = 0;

                double px = testStartPosition[0];
                double pz = testStartPosition[2];
                double qx = testFrame1Position[0];
                double qz = testFrame1Position[2];

                bool intOnSquishEdge[2] = { false, false };

                for (int i = 0; i < 3; i++) {
                    double ax = startTriangles[f][i][0];
                    double ay = startTriangles[f][i][1];
                    double az = startTriangles[f][i][2];
                    double bx = startTriangles[f][(i + 1) % 3][0];
                    double by = startTriangles[f][(i + 1) % 3][1];
                    double bz = startTriangles[f][(i + 1) % 3][2];

                    double t = ((pz - qz) * (pz - az) + (px - qx) * (px - ax)) / ((pz - qz) * (bz - az) + (px - qx) * (bx - ax));

                    if (t >= 0.0 && t <= 1.0) {
                        intOnSquishEdge[intersections] = (f == 0 && ((i == 0 && squishCeilings[2]) || (i == 1 && squishCeilings[0]))) || (f == 1 && ((i == 1 && squishCeilings[1]) || (i == 2 && squishCeilings[3])));
                        intersectionPoints[intersections][0] = ax + (bx - ax) * t;
                        intersectionPoints[intersections][1] = ay + (by - ay) * t;
                        intersectionPoints[intersections][2] = az + (bz - az) * t;
                        intersections++;
                    }
                }

                double cutPoints[2];

                double ax = intersectionPoints[0][0];
                double ay = intersectionPoints[0][1];
                double az = intersectionPoints[0][2];
                double bx = intersectionPoints[1][0];
                double by = intersectionPoints[1][1];
                double bz = intersectionPoints[1][2];

                px = testFrame1Position[0];
                pz = testFrame1Position[2];

                int angleIdx = gReverseArctanTable[angle];
                int prevAngle = (65536 + gArctanTableG[(angleIdx + 8191) % 8192]) % 65536;
                int nextAngle = (65536 + gArctanTableG[(angleIdx + 1) % 8192]) % 65536;

                double m = (gSineTableG[nextAngle >> 4] + gSineTableG[angle >> 4]) / 2.0;
                double n = (gCosineTableG[nextAngle >> 4] + gCosineTableG[angle >> 4]) / 2.0;

                cutPoints[0] = ((pz - az) + (n / m) * (ax - px)) / ((bz - az) - (n / m) * (bx - ax));

                m = (gSineTableG[prevAngle >> 4] + gSineTableG[angle >> 4]) / 2.0;
                n = (gCosineTableG[prevAngle >> 4] + gCosineTableG[angle >> 4]) / 2.0;

                cutPoints[1] = ((pz - az) + (n / m) * (ax - px)) / ((bz - az) - (n / m) * (bx - ax));

                if (cutPoints[0] > cutPoints[1]) {
                    double temp = cutPoints[0];
                    cutPoints[0] = cutPoints[1];
                    cutPoints[1] = temp;
                }

                cutPoints[0] = fmax(cutPoints[0], 0.0);
                cutPoints[1] = fmin(cutPoints[1], 1.0);

                if (cutPoints[0] <= cutPoints[1]) {
                    intersectionPoints[0][0] = ax + (bx - ax) * cutPoints[0];
                    intersectionPoints[0][1] = ay + (by - ay) * cutPoints[0];
                    intersectionPoints[0][2] = az + (bz - az) * cutPoints[0];

                    intersectionPoints[1][0] = ax + (bx - ax) * cutPoints[1];
                    intersectionPoints[1][1] = ay + (by - ay) * cutPoints[1];
                    intersectionPoints[1][2] = az + (bz - az) * cutPoints[1];

                    bool foundStartPosition = false;

                    float marioMinY = fmaxf(-2971.0f, testOneUpPlatformPosition[1] - 78.0f);

                    float endHeight;
                    int floorIdx;

                    if (q1 == 1) {
                        if (fmaxf(intersectionPoints[0][1], intersectionPoints[1][1]) > marioMinY && fminf(intersectionPoints[0][1], intersectionPoints[1][1]) < testOneUpPlatformPosition[1]) {
                            if (intersectionPoints[0][1] < marioMinY) {
                                double ratio = (intersectionPoints[1][1] == intersectionPoints[0][1]) ? 0.0 : (marioMinY - intersectionPoints[0][1]) / (intersectionPoints[1][1] - intersectionPoints[0][1]);
                                cutPoints[0] = cutPoints[0] + (cutPoints[1] - cutPoints[0]) * ratio;
                                intersectionPoints[0][0] = intersectionPoints[0][0] + (intersectionPoints[1][0] - intersectionPoints[0][0]) * ratio;
                                intersectionPoints[0][2] = intersectionPoints[0][2] + (intersectionPoints[1][2] - intersectionPoints[0][2]) * ratio;
                                intersectionPoints[0][1] = marioMinY;
                            }
                            else if (intersectionPoints[1][1] < marioMinY) {
                                double ratio = (intersectionPoints[0][1] == intersectionPoints[1][1]) ? 0.0 : (marioMinY - intersectionPoints[1][1]) / (intersectionPoints[0][1] - intersectionPoints[1][1]);
                                cutPoints[1] = cutPoints[1] + (cutPoints[0] - cutPoints[1]) * ratio;
                                intersectionPoints[1][0] = intersectionPoints[1][0] + (intersectionPoints[0][0] - intersectionPoints[1][0]) * ratio;
                                intersectionPoints[1][2] = intersectionPoints[1][2] + (intersectionPoints[0][2] - intersectionPoints[1][2]) * ratio;
                                intersectionPoints[1][1] = marioMinY;
                            }

                            if (intersectionPoints[0][1] > testOneUpPlatformPosition[1]) {
                                double ratio = (intersectionPoints[1][1] == intersectionPoints[0][1]) ? 0.0 : (testOneUpPlatformPosition[1] - intersectionPoints[0][1]) / (intersectionPoints[1][1] - intersectionPoints[0][1]);
                                cutPoints[0] = cutPoints[0] + (cutPoints[1] - cutPoints[0]) * ratio;
                                intersectionPoints[0][0] = intersectionPoints[0][0] + (intersectionPoints[1][0] - intersectionPoints[0][0]) * ratio;
                                intersectionPoints[0][2] = intersectionPoints[0][2] + (intersectionPoints[1][2] - intersectionPoints[0][2]) * ratio;
                                intersectionPoints[0][1] = testOneUpPlatformPosition[1];
                            }
                            else if (intersectionPoints[1][1] < testOneUpPlatformPosition[1]) {
                                double ratio = (intersectionPoints[0][1] == intersectionPoints[1][1]) ? 0.0 : (testOneUpPlatformPosition[1] - intersectionPoints[1][1]) / (intersectionPoints[0][1] - intersectionPoints[1][1]);
                                cutPoints[1] = cutPoints[1] + (cutPoints[0] - cutPoints[1]) * ratio;
                                intersectionPoints[1][0] = intersectionPoints[1][0] + (intersectionPoints[0][0] - intersectionPoints[1][0]) * ratio;
                                intersectionPoints[1][2] = intersectionPoints[1][2] + (intersectionPoints[0][2] - intersectionPoints[1][2]) * ratio;
                                intersectionPoints[1][1] = testOneUpPlatformPosition[1];
                            }

                            testStartPosition[0] = (intersectionPoints[0][0] + intersectionPoints[1][0]) / 2.0f;
                            testStartPosition[2] = (intersectionPoints[0][2] + intersectionPoints[1][2]) / 2.0f;

                            float floorHeight;
                            floorIdx = find_floor_triangles(testStartPosition, startTriangles, startNormals, &floorHeight);
                            testStartPosition[1] = floorHeight;
                            endHeight = floorHeight;

                            foundStartPosition = true;
                        }
                    }
                    else {
                        const int nTestFloors = 10;

                        float testFloors[nTestFloors][6] = {
                            { -4453.0f, -4146.0f, -613.0f, 307.0f, -2661.0f, 1.0f },
                            { -4453.0f, -4146.0f, -306.0f, 307.0f, -2743.0f, 1.0f },
                            { 3072.0f, 4301.0f, -40.0f, 881.0f, -2764.0f, 1.0f },
                            { 2048.0f, 2662.0f, -347.0f, 881.0f, -2764.0f, 1.0f },
                            { 2661.0f, 3072.0f, 61.0f, 471.0f, -2764.0f, 1.0f },
                            { -7986.0f, -7064.0f, -511.0f, 512.0f, -2764.0f, 1.0f },
                            { -306.0f, 0.0f, 307.0f, 922.0f, -2866.0f, 1.0f },
                            { -6553.0f, -6041.0f, -306.0f, 307.0f, -2866.0f, 1.0f },
                            { 5222.0f, 6298.0f, -40.0f, 573.0f, -2917.0f, 1.0f },
                            { -6041.0f, -306.0f, -306.0f, 307.0f, returnPosition[1] - 1.0f, 1.0f}
                        };

                        float intendedPositions[2][3];
                        intendedPositions[0][0] = intersectionPoints[0][0];
                        intendedPositions[0][1] = intersectionPoints[0][1];
                        intendedPositions[0][2] = intersectionPoints[0][2];
                        intendedPositions[1][0] = intersectionPoints[1][0];
                        intendedPositions[1][1] = intersectionPoints[1][1];
                        intendedPositions[1][2] = intersectionPoints[1][2];

                        float xVel0s[2] = { (testFrame1Position[0] - intersectionPoints[0][0]) / (startNormals[f][1] + (q1 - 1)), (testFrame1Position[0] - intersectionPoints[1][0]) / (startNormals[f][1] + (q1 - 1)) };
                        float zVel0s[2] = { (testFrame1Position[2] - intersectionPoints[0][2]) / (startNormals[f][1] + (q1 - 1)), (testFrame1Position[2] - intersectionPoints[1][2]) / (startNormals[f][1] + (q1 - 1)) };

                        float currentNormalY = startNormals[f][1];

                        bool foundFloor = true;

                        for (int q = 1; foundFloor && q < q1; q++) {
                            foundFloor = false;
                            intendedPositions[0][0] = intendedPositions[0][0] + currentNormalY * (xVel0s[0] / 4.0);
                            intendedPositions[0][2] = intendedPositions[0][2] + currentNormalY * (zVel0s[0] / 4.0);
                            intendedPositions[1][0] = intendedPositions[1][0] + currentNormalY * (xVel0s[1] / 4.0);
                            intendedPositions[1][2] = intendedPositions[1][2] + currentNormalY * (zVel0s[1] / 4.0);

                            for (int i = 0; i < nTestFloors; i++) {
                                float testHeight = testFloors[i][4];

                                double yMin = testHeight - 78.0f;
                                double yMax = testHeight + 100.0f;

                                double yMinT = (yMin - intendedPositions[0][1]) / (intendedPositions[1][1] - intendedPositions[0][1]);
                                double yMaxT = (yMax - intendedPositions[0][1]) / (intendedPositions[1][1] - intendedPositions[0][1]);

                                double minT = fmax(0.0, fmin(yMinT, yMaxT));
                                double maxT = fmin(1.0, fmax(yMinT, yMaxT));

                                if (minT <= maxT) {
                                    double xMin = testFloors[i][0];
                                    double xMax = testFloors[i][1];
                                    double zMin = testFloors[i][2];
                                    double zMax = testFloors[i][3];

                                    double xMinT = (xMin - intendedPositions[0][0]) / (intendedPositions[1][0] - intendedPositions[0][0]);
                                    double xMaxT = (xMax - intendedPositions[0][0]) / (intendedPositions[1][0] - intendedPositions[0][0]);

                                    double zMinT = (zMin - intendedPositions[0][2]) / (intendedPositions[1][2] - intendedPositions[0][2]);
                                    double zMaxT = (zMax - intendedPositions[0][2]) / (intendedPositions[1][2] - intendedPositions[0][2]);

                                    minT = fmax(minT, fmax(fmin(xMinT, xMaxT), fmin(zMinT, zMaxT)));
                                    maxT = fmin(maxT, fmin(fmax(xMinT, xMaxT), fmax(zMinT, zMaxT)));

                                    if (minT <= maxT) {
                                        float ipX1 = minT * (intendedPositions[1][0] - intendedPositions[0][0]) + intendedPositions[0][0];
                                        float ipZ1 = minT * (intendedPositions[1][2] - intendedPositions[0][2]) + intendedPositions[0][2];
                                        float ipX2 = maxT * (intendedPositions[1][0] - intendedPositions[0][0]) + intendedPositions[0][0];
                                        float ipZ2 = maxT * (intendedPositions[1][2] - intendedPositions[0][2]) + intendedPositions[0][2];

                                        intendedPositions[0][0] = ipX1;
                                        intendedPositions[0][1] = testHeight;
                                        intendedPositions[0][2] = ipZ1;
                                        intendedPositions[1][0] = ipX2;
                                        intendedPositions[1][1] = testHeight;
                                        intendedPositions[1][2] = ipZ2;

                                        currentNormalY = testFloors[i][5];

                                        ipX1 = minT * (intersectionPoints[1][0] - intersectionPoints[0][0]) + intersectionPoints[0][0];
                                        ipZ1 = minT * (intersectionPoints[1][2] - intersectionPoints[0][2]) + intersectionPoints[0][2];
                                        ipX2 = maxT * (intersectionPoints[1][0] - intersectionPoints[0][0]) + intersectionPoints[0][0];
                                        ipZ2 = maxT * (intersectionPoints[1][2] - intersectionPoints[0][2]) + intersectionPoints[0][2];

                                        float ipY1 = minT * (intersectionPoints[1][1] - intersectionPoints[0][1]) + intersectionPoints[0][1];
                                        float ipY2 = maxT * (intersectionPoints[1][1] - intersectionPoints[0][1]) + intersectionPoints[0][1];

                                        intersectionPoints[0][0] = ipX1;
                                        intersectionPoints[0][1] = ipY1;
                                        intersectionPoints[0][2] = ipZ1;
                                        intersectionPoints[1][0] = ipX2;
                                        intersectionPoints[1][1] = ipY2;
                                        intersectionPoints[1][2] = ipZ2;

                                        foundFloor = true;
                                        endHeight = testHeight;
                                        break;
                                    }
                                }
                            }
                        }

                        if (foundFloor) {
                            testStartPosition[0] = (intersectionPoints[0][0] + intersectionPoints[1][0]) / 2.0f;
                            testStartPosition[2] = (intersectionPoints[0][2] + intersectionPoints[1][2]) / 2.0f;

                            float floorHeight;
                            floorIdx = find_floor_triangles(testStartPosition, startTriangles, startNormals, &floorHeight);
                            testStartPosition[1] = floorHeight;

                            foundStartPosition = true;
                        }
                    }

                    if (foundStartPosition) {
                        //if ((intOnEdge[0] && cutPoints[0] == 0.0) || (intOnEdge[1] && cutPoints[0] == 1.0)

                        if (floorIdx == f && endHeight > marioMinY && endHeight < testOneUpPlatformPosition[1]) {
                            testFrame1Position[1] = endHeight;

                            foundSolution = true;
                            struct PUSolution puSol = puSolutions[solIdx];
                            struct UpwarpSolution uwSol = upwarpSolutions[puSol.upwarpSolutionIdx];
                            struct PlatformSolution platSol = platSolutions[uwSol.platformSolutionIdx];
                            printf("---------------------------------------\nFound Solution:\n---------------------------------------\n    Start Position: %.10g, %.10g, %.10g\n    Frame 1 Position: %.10g, %.10g, %.10g\n    Frame 2 Position: %.10g, %.10g, %.10g\n    Return Position: %.10g, %.10g, %.10g\n    PU Route Speed: %.10g (x=%.10g, z=%.10g)\n    PU Return Speed: %.10g (x=%.10g, z=%.10g)\n    Frame 1 Q-steps: %d\n    Frame 2 Q-steps: %d\n    Frame 3 Q-steps: %d\n", testStartPosition[0], testStartPosition[1], testStartPosition[2], testFrame1Position[0], testFrame1Position[1], testFrame1Position[2], testOneUpPlatformPosition[0], testOneUpPlatformPosition[1], testOneUpPlatformPosition[2], returnPosition[0], returnPosition[1], returnPosition[2], vel1, xVel1, zVel1, endSpeed, xVel2a, zVel2a, q1, q2, q3);
                            printf("    10k Stick X: %d\n    10k Stick Y: %d\n    10k Camera Yaw: %d\n    Start Floor Normal: %.10g, %.10g, %.10g\n    Start Position Limit 1: %.10g %.10g %.10g\n    Start Position Limit 2: %.10g %.10g %.10g\n", trueX, trueY, cameraYaw, startNormals[f][0], startNormals[f][1], startNormals[f][2], intersectionPoints[0][0], intersectionPoints[0][1], intersectionPoints[0][2], intersectionPoints[1][0], intersectionPoints[1][1], intersectionPoints[1][2]);
                            printf("---------------------------------------\n    Tilt Frames: %d\n    Post-Tilt Platform Normal: %.10g, %.10g, %.10g\n    Post-Tilt Position: %.10g, %.10g, %.10g\n    Post-Upwarp Position: %.10g, %.10g, %.10g\n    Upwarp PU X: %d\n    Upwarp PU Z: %d\n    Upwarp Slide Facing Angle: %d\n    Upwarp Slide Intended Mag: %.10g\n    Upwarp Slide Intended DYaw: %d\n---------------------------------------\n\n\n", platSol.nFrames, platSol.endNormal[0], platSol.endNormal[1], platSol.endNormal[2], platSol.endPosition[0], platSol.endPosition[1], platSol.endPosition[2], uwSol.upwarpPosition[0], uwSol.upwarpPosition[1], uwSol.upwarpPosition[2], uwSol.pux, uwSol.puz, puSol.angle, puSol.stickMag, puSol.intendedDYaw);

                            int idx = atomicAdd(&n10KSolutionsNonHAU, 1);

                            if (idx < MAX_10K_SOLUTIONS_NON_HAU) {
                                struct TenKSolutionNonHAU solution;
                                solution.puSolutionIdx = solIdx;
                                solution.startFloorIdx = f;
                                solution.startPosition[0] = testStartPosition[0];
                                solution.startPosition[1] = testStartPosition[1];
                                solution.startPosition[2] = testStartPosition[2];
                                solution.startPositionLimits[0][0] = intersectionPoints[0][0];
                                solution.startPositionLimits[0][1] = intersectionPoints[0][1];
                                solution.startPositionLimits[0][2] = intersectionPoints[0][2];
                                solution.startPositionLimits[1][0] = intersectionPoints[1][0];
                                solution.startPositionLimits[1][1] = intersectionPoints[1][1];
                                solution.startPositionLimits[1][2] = intersectionPoints[1][2];
                                solution.frame1Position[0] = testFrame1Position[0];
                                solution.frame1Position[1] = testFrame1Position[1];
                                solution.frame1Position[2] = testFrame1Position[2];
                                solution.frame2Position[0] = testOneUpPlatformPosition[0];
                                solution.frame2Position[1] = testOneUpPlatformPosition[1];
                                solution.frame2Position[2] = testOneUpPlatformPosition[2];
                                solution.frame1QSteps = q1;
                                solution.frame2QSteps = q2;
                                solution.frame3QSteps = q3;
                                solution.pre10Kspeed = vel1;
                                solution.pre10KVel[0] = xVel1;
                                solution.pre10KVel[1] = zVel1;
                                solution.returnVel[0] = xVel2a;
                                solution.returnVel[1] = zVel2a;
                                solution.stick10K[0] = trueX;
                                solution.stick10K[1] = trueY;
                                solution.cameraYaw10K = cameraYaw;
                                tenKSolutionsNonHAU[idx] = solution;
                            }
                        }
                    }
                }
            }
        }
    }

    return foundSolution;
}

__device__ bool test_one_up_position(int solIdx, float* startPosition, float* oneUpPlatformPosition, float* returnPosition, float endSpeed, float oneUpPlatformXMin, float oneUpPlatformXMax, float oneUpPlatformYMin, float oneUpPlatformYMax, float oneUpPlatformZMin, float oneUpPlatformZMax, float oneUpPlatformNormalX, float oneUpPlatformNormalY, int f, int q3, int minQ1, int maxQ1, int minQ2, int maxQ2) {
    float cameraPositions[4][3] = { {-8192, -2918, -8192}, {-8192, -2918, 8191}, {8191, -2918, -8192}, {8191, -2918, 8191} };
    bool foundSolution = false;

    const int maxF2AngleChange = 552;

    int minCameraYaw = 0;
    int maxCameraYaw = 0;

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

        float xVel2 = 4.0f * (returnPosition[0] - oneUpPlatformPosition[0]) / (oneUpPlatformNormalY + (q3 - 1));
        float zVel2 = 4.0f * (returnPosition[2] - oneUpPlatformPosition[2]) / (oneUpPlatformNormalY + (q3 - 1));

        float s = xVel2 / zVel2;

        int angle = atan2sG(-zVel2, -xVel2);
        angle = (65536 + angle) % 65536;

        int closestAngle = atan2sG(-zVel2, -xVel2);
        closestAngle = (65536 + closestAngle) % 65536;

        int minFacingAngleIdx = gReverseArctanTable[closestAngle];
        int maxFacingAngleIdx = gReverseArctanTable[closestAngle];

        while ((65536 + closestAngle - gArctanTableG[(minFacingAngleIdx + 8192) % 8192]) % 65536 < maxF2AngleChange) {
            minFacingAngleIdx = minFacingAngleIdx - 1;
        }

        while ((65536 + gArctanTableG[(maxFacingAngleIdx + 1) % 8192] - closestAngle) % 65536 < maxF2AngleChange) {
            maxFacingAngleIdx = maxFacingAngleIdx + 1;
        }

        for (int angleIdx = minFacingAngleIdx; angleIdx <= maxFacingAngleIdx; angleIdx++) {
            int angle = gArctanTableG[(8192 + angleIdx) % 8192];

            for (int q2 = minQ2; q2 <= maxQ2; q2++) {
                for (int q1 = minQ1; q1 <= maxQ1; q1++) {
                    double eqA = ((double)q2 * (double)q2 - ((double)startNormals[f][1] + (q1 - 1)) * ((double)startNormals[f][1] + (q1 - 1))) / 16.0;
                    double eqB = ((double)q2 / 2.0) * ((startPosition[0] - oneUpPlatformPosition[0]) * gSineTableG[angle >> 4] + (startPosition[2] - oneUpPlatformPosition[2]) * gCosineTableG[angle >> 4]);
                    double eqC = ((startPosition[0] - oneUpPlatformPosition[0]) * (startPosition[0] - oneUpPlatformPosition[0]) + (startPosition[2] - oneUpPlatformPosition[2]) * (startPosition[2] - oneUpPlatformPosition[2]));
                    double eqDet = (eqB * eqB) - (4.0 * eqA * eqC);

                    if (eqB < 0 && eqDet >= 0) {
                        float vel1 = (-eqB - sqrt(eqDet)) / (2.0 * eqA);

                        float xVel1 = vel1 * gSineTableG[angle >> 4];
                        float zVel1 = vel1 * gCosineTableG[angle >> 4];

                        float frame1Position[3] = { oneUpPlatformPosition[0], startPosition[1], oneUpPlatformPosition[2] };
                        bool inBoundsTest = true;

                        for (int q = 0; q < q2; q++) {
                            frame1Position[0] = frame1Position[0] - (xVel1 / 4.0f);
                            frame1Position[2] = frame1Position[2] - (zVel1 / 4.0f);

                            if (!check_inbounds(frame1Position)) {
                                inBoundsTest = false;
                                break;
                            }
                        }

                        if (inBoundsTest) {
                            int angle2 = atan2sG(frame1Position[2] - startPosition[2], frame1Position[0] - startPosition[0]);
                            angle2 = (65536 + angle2) % 65536;

                            if (angle == angle2) {
                                double m = (double)endSpeed / (double)vel1;
                                double m1 = 32.0 * ((m - 0.92) / 0.02) / (double)(0.5f + (0.5f * vel1 / 100.0f));

                                double t = (double)xVel1 / (double)zVel1;

                                double n;

                                if (zVel2 == 0) {
                                    n = zVel1 / xVel2;
                                }
                                else if (zVel1 == 0) {
                                    n = -zVel2 / xVel1;
                                }
                                else if (xVel2 == 0) {
                                    n = -t;
                                }
                                else {
                                    bool signTest = (zVel1 > 0 && zVel2 > 0) || (zVel1 < 0 && zVel2 < 0);

                                    if (signTest) {
                                        n = (-((double)s * (double)t) - 1.0 + sqrt(((double)s * (double)t - 1.0) * ((double)s * (double)t - 1.0) + 4.0 * (double)s * (double)s)) / (2.0 * (double)s);
                                    }
                                    else {
                                        n = (-((double)s * (double)t) - 1.0 - sqrt(((double)s * (double)t - 1.0) * ((double)s * (double)t - 1.0) + 4.0 * (double)s * (double)s)) / (2.0 * (double)s);
                                    }
                                }

                                double n1 = 32.0 * n / 0.05;

                                double targetDYaw = 65536.0 * (atan2(n1, m1) / (2.0 * M_PI));
                                double targetMag = sqrtf(m1 * m1 + n1 * n1);

                                double stickAngle = fmod(65536.0 + fmod(targetDYaw + angle - cameraYaw, 65536.0), 65536.0);
                                double stickMagnitude = sqrt(128.0 * targetMag);

                                if (stickMagnitude < 70.0) {
                                    if (stickMagnitude < 64.0) {
                                        double yS = -stickMagnitude * cos(2.0 * M_PI * (stickAngle / 65536));
                                        double xS = stickMagnitude * sin(2.0 * M_PI * (stickAngle / 65536));

                                        int x = round(xS);
                                        int y = round(yS);

                                        if (x != -1 && x != 1 && y != -1 && y != 1) {
                                            if (test_stick_position(solIdx, x, y, endSpeed, vel1, xVel1, zVel1, angle, cameraYaw, startPosition, oneUpPlatformPosition, oneUpPlatformXMin, oneUpPlatformXMax, oneUpPlatformYMin, oneUpPlatformYMax, oneUpPlatformZMin, oneUpPlatformZMax, oneUpPlatformNormalX, oneUpPlatformNormalY, f, frame1Position, returnPosition, q1, q2, q3)) {
                                                foundSolution = true;
                                            }
                                        }
                                    }
                                    else {
                                        double yS = -64.0 * sin(2.0 * M_PI * (stickAngle / 65536));
                                        double xS = 64.0 * cos(2.0 * M_PI * (stickAngle / 65536));

                                        if (fabs(xS) > fabs(yS)) {
                                            int minX = (fabs(yS) < 0.00001) ? ((xS < 0) ? -128 : 64) : ((xS < 0) ? floor(xS) : ceil(xS));
                                            int maxX = (fabs(yS) < 0.00001) ? ((xS < 0) ? -64 : 127) : ((xS < 0) ? ceil(-128 * xS / yS) : floor(127 * xS / yS));

                                            for (int x = minX; x <= maxX; x++) {
                                                double y = (double)x * (yS / xS);

                                                if (fabs(floor(y)) != 1.0) {
                                                    if (test_stick_position(solIdx, x, floor(y), endSpeed, vel1, xVel1, zVel1, angle, cameraYaw, startPosition, oneUpPlatformPosition, oneUpPlatformXMin, oneUpPlatformXMax, oneUpPlatformYMin, oneUpPlatformYMax, oneUpPlatformZMin, oneUpPlatformZMax, oneUpPlatformNormalX, oneUpPlatformNormalY, f, frame1Position, returnPosition, q1, q2, q3)) {
                                                        foundSolution = true;
                                                    }
                                                }

                                                if (fabs(ceil(y)) != 1.0) {
                                                    if (test_stick_position(solIdx, x, ceil(y), endSpeed, vel1, xVel1, zVel1, angle, cameraYaw, startPosition, oneUpPlatformPosition, oneUpPlatformXMin, oneUpPlatformXMax, oneUpPlatformYMin, oneUpPlatformYMax, oneUpPlatformZMin, oneUpPlatformZMax, oneUpPlatformNormalX, oneUpPlatformNormalY, f, frame1Position, returnPosition, q1, q2, q3)) {
                                                        foundSolution = true;
                                                    }
                                                }
                                            }
                                        }
                                        else {
                                            int minY = (fabs(xS) < 0.00001) ? ((yS < 0) ? -128 : 64) : ((yS < 0) ? floor(yS) : ceil(yS));
                                            int maxY = (fabs(xS) < 0.00001) ? ((yS < 0) ? -64 : 127) : ((yS < 0) ? ceil(-128 * yS / xS) : floor(127 * yS / xS));

                                            for (int y = minY; y <= maxY; y++) {
                                                double x = (double)y * (xS / yS);

                                                if (fabs(floor(x)) != 1.0) {
                                                    if (test_stick_position(solIdx, floor(x), y, endSpeed, vel1, xVel1, zVel1, angle, cameraYaw, startPosition, oneUpPlatformPosition, oneUpPlatformXMin, oneUpPlatformXMax, oneUpPlatformYMin, oneUpPlatformYMax, oneUpPlatformZMin, oneUpPlatformZMax, oneUpPlatformNormalX, oneUpPlatformNormalY, f, frame1Position, returnPosition, q1, q2, q3)) {
                                                        foundSolution = true;
                                                    }
                                                }

                                                if (fabs(ceil(x)) != 1.0) {
                                                    if (test_stick_position(solIdx, ceil(x), y, endSpeed, vel1, xVel1, zVel1, angle, cameraYaw, startPosition, oneUpPlatformPosition, oneUpPlatformXMin, oneUpPlatformXMax, oneUpPlatformYMin, oneUpPlatformYMax, oneUpPlatformZMin, oneUpPlatformZMax, oneUpPlatformNormalX, oneUpPlatformNormalY, f, frame1Position, returnPosition, q1, q2, q3)) {
                                                        foundSolution = true;
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
        }
    }

    return foundSolution;
}

__device__ bool find_10k_route(int solIdx, int f, int d, int h, int e, int q3, int minQ1, int maxQ1, int minQ2, int maxQ2) {
    struct PlatformSolution* sol = &(platSolutions[upwarpSolutions[puSolutions[solIdx].upwarpSolutionIdx].platformSolutionIdx]);
    float returnSpeed = puSolutions[solIdx].returnSpeed;

    bool foundSolution = false;

    float startPosition[3];
    startPosition[0] = ((float)startTriangles[f][0][0] + (float)startTriangles[f][1][0] + (float)startTriangles[f][2][0]) / 3.0;
    startPosition[1] = ((float)startTriangles[f][0][1] + (float)startTriangles[f][1][1] + (float)startTriangles[f][2][1]) / 3.0;
    startPosition[2] = ((float)startTriangles[f][0][2] + (float)startTriangles[f][1][2] + (float)startTriangles[f][2][2]) / 3.0;

    double signX = d == 0 ? -1.0 : 1.0;
    double signZ = h == 0 ? -1.0 : 1.0;

    float oneUpPlatformNormalY = (d == 0) ? oneUpPlatformNormalYRight : oneUpPlatformNormalYLeft;
    float oneUpPlatformNormalX = (d == 0) ? oneUpPlatformNormalXRight : oneUpPlatformNormalXLeft;
    float oneUpPlatformXMin = (d == 0) ? oneUpPlatformXMinRight : oneUpPlatformXMinLeft;
    float oneUpPlatformXMax = (d == 0) ? oneUpPlatformXMaxRight : oneUpPlatformXMaxLeft;
    float oneUpPlatformYMin = (d == 0) ? oneUpPlatformYMinRight : oneUpPlatformYMinLeft;
    float oneUpPlatformYMax = (d == 0) ? oneUpPlatformYMaxRight : oneUpPlatformYMaxLeft;
    float oneUpPlatformZMin = (d == 0) ? oneUpPlatformZMinRight : oneUpPlatformZMinLeft;
    float oneUpPlatformZMax = (d == 0) ? oneUpPlatformZMaxRight : oneUpPlatformZMaxLeft;

    double r = fabs((double)returnSpeed * (double)(oneUpPlatformNormalY + (q3 - 1))) / 4.0;

    if (e == 0) {
        int maxXPU = (d == 0) ? (int)ceil((sol->returnPosition[0] - (r * sqrt(0.5)) - oneUpPlatformXMax) / 65536.0) : (int)floor((sol->returnPosition[0] + r - oneUpPlatformXMin) / 65536.0);
        int minXPU = (d == 0) ? (int)ceil((sol->returnPosition[0] - r - oneUpPlatformXMax) / 65536.0) : (int)floor((sol->returnPosition[0] + (r * sqrt(0.5)) - oneUpPlatformXMin) / 65536.0);

        for (int i = minXPU; i <= maxXPU; i++) {
            double z0 = signZ * sqrt(r * r - (i * 65536.0 + oneUpPlatformXMin - sol->returnPosition[0]) * (i * 65536.0 + oneUpPlatformXMin - sol->returnPosition[0])) + sol->returnPosition[2];
            double z1 = signZ * sqrt(r * r - (i * 65536.0 + oneUpPlatformXMax - sol->returnPosition[0]) * (i * 65536.0 + oneUpPlatformXMax - sol->returnPosition[0])) + sol->returnPosition[2];

            int minZPU = (int)ceil((fmin(z0, z1) - oneUpPlatformZMax) / 65536.0);
            minZPU = (h == 0) ? minZPU : max(1, minZPU);
            int maxZPU = (int)floor((fmax(z0, z1) - oneUpPlatformZMin) / 65536.0);
            maxZPU = (h == 0) ? min(0, maxZPU) : maxZPU;

            for (int j = minZPU; j <= maxZPU; j++) {
                double x0 = signX * sqrt(r * r - (j * 65536.0 + oneUpPlatformZMin - sol->returnPosition[2]) * (j * 65536.0 + oneUpPlatformZMin - sol->returnPosition[2])) + sol->returnPosition[0];
                double x1 = signX * sqrt(r * r - (j * 65536.0 + oneUpPlatformZMax - sol->returnPosition[2]) * (j * 65536.0 + oneUpPlatformZMax - sol->returnPosition[2])) + sol->returnPosition[0];

                double minX = fmax(fmin(x0, x1), i * 65536.0 + oneUpPlatformXMin);
                double maxX = fmin(fmax(x0, x1), i * 65536.0 + oneUpPlatformXMax);

                double minXY = (double)(oneUpPlatformYMax - oneUpPlatformYMin) * (minX - (65536.0 * i) - oneUpPlatformXMin) / (double)(oneUpPlatformXMax - oneUpPlatformXMin) + oneUpPlatformYMin;
                double maxXY = (double)(oneUpPlatformYMax - oneUpPlatformYMin) * (maxX - (65536.0 * i) - oneUpPlatformXMin) / (double)(oneUpPlatformXMax - oneUpPlatformXMin) + oneUpPlatformYMin;

                double minXZ = signZ * sqrt(r * r - (minX - sol->returnPosition[0]) * (minX - sol->returnPosition[0])) + sol->returnPosition[2];
                double maxXZ = signZ * sqrt(r * r - (maxX - sol->returnPosition[0]) * (maxX - sol->returnPosition[0])) + sol->returnPosition[2];

                float oneUpPlatformPosition[3] = { (maxX + minX) / 2.0, (maxXY + minXY) / 2.0, (maxXZ + minXZ) / 2.0 };

                if (test_one_up_position(solIdx, startPosition, oneUpPlatformPosition, sol->returnPosition, returnSpeed, oneUpPlatformXMin, oneUpPlatformXMax, oneUpPlatformYMin, oneUpPlatformYMax, oneUpPlatformZMin, oneUpPlatformZMax, oneUpPlatformNormalX, oneUpPlatformNormalY, f, q3, minQ1, maxQ1, minQ2, maxQ2)) {
                    foundSolution = true;
                }
            }
        }
    }
    else {
        int maxZPU = (h == 0) ? (int)ceil((sol->returnPosition[2] - (r * sqrt(0.5)) - oneUpPlatformZMax) / 65536.0) : (int)floor((sol->returnPosition[2] + r - oneUpPlatformZMin) / 65536.0);
        int minZPU = (h == 0) ? (int)ceil((sol->returnPosition[2] - r - oneUpPlatformZMax) / 65536.0) : (int)floor((sol->returnPosition[2] + (r * sqrt(0.5)) - oneUpPlatformZMin) / 65536.0);

        for (int i = minZPU; i <= maxZPU; i++) {
            double x0 = signX * sqrt(r * r - (i * 65536.0 + oneUpPlatformZMin - sol->returnPosition[2]) * (i * 65536.0 + oneUpPlatformZMin - sol->returnPosition[2])) + sol->returnPosition[0];
            double x1 = signX * sqrt(r * r - (i * 65536.0 + oneUpPlatformZMax - sol->returnPosition[2]) * (i * 65536.0 + oneUpPlatformZMax - sol->returnPosition[2])) + sol->returnPosition[0];

            int minXPU = (int)ceil((fmin(x0, x1) - oneUpPlatformXMax) / 65536.0);
            minXPU = (d == 0) ? minXPU : max(1, minXPU);
            int maxXPU = (int)floor((fmax(x0, x1) - oneUpPlatformXMin) / 65536.0);
            maxXPU = (d == 0) ? min(0, maxXPU) : maxXPU;

            for (int j = minXPU; j <= maxXPU; j++) {
                double z0 = signZ * sqrt(r * r - (j * 65536.0 + oneUpPlatformXMin - sol->returnPosition[0]) * (j * 65536.0 + oneUpPlatformXMin - sol->returnPosition[0])) + sol->returnPosition[2];
                double z1 = signZ * sqrt(r * r - (j * 65536.0 + oneUpPlatformXMax - sol->returnPosition[0]) * (j * 65536.0 + oneUpPlatformXMax - sol->returnPosition[0])) + sol->returnPosition[2];

                double minZ = fmax(fmin(z0, z1), i * 65536.0 + oneUpPlatformZMin);
                double maxZ = fmin(fmax(z0, z1), i * 65536.0 + oneUpPlatformZMax);

                double minZX = signX * sqrt(r * r - (minZ - sol->returnPosition[2]) * (minZ - sol->returnPosition[2])) + sol->returnPosition[0];
                double maxZX = signX * sqrt(r * r - (maxZ - sol->returnPosition[2]) * (maxZ - sol->returnPosition[2])) + sol->returnPosition[0];

                double minZY = (double)(oneUpPlatformYMax - oneUpPlatformYMin) * (minZX - (65536.0 * i) - oneUpPlatformXMin) / (double)(oneUpPlatformXMax - oneUpPlatformXMin) + oneUpPlatformYMin;
                double maxZY = (double)(oneUpPlatformYMax - oneUpPlatformYMin) * (maxZX - (65536.0 * i) - oneUpPlatformXMin) / (double)(oneUpPlatformXMax - oneUpPlatformXMin) + oneUpPlatformYMin;

                float oneUpPlatformPosition[3] = { (maxZX + minZX) / 2.0, (maxZY + minZY) / 2.0, (maxZ + minZ) / 2.0 };

                if (test_one_up_position(solIdx, startPosition, oneUpPlatformPosition, sol->returnPosition, returnSpeed, oneUpPlatformXMin, oneUpPlatformXMax, oneUpPlatformYMin, oneUpPlatformYMax, oneUpPlatformZMin, oneUpPlatformZMax, oneUpPlatformNormalX, oneUpPlatformNormalY, f, q3, minQ1, maxQ1, minQ2, maxQ2)) {
                    foundSolution = true;
                }
            }
        }
    }

    return foundSolution;
}

__global__ void test_pu_solution(int q3, int minQ1, int maxQ1, int minQ2, int maxQ2) {
    long long idx = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;

    if (idx < (long long)16 * (long long)nPUSolutions) {
        int solIdx = idx % nPUSolutions;
        idx = idx / nPUSolutions;
        int f = idx % 2;
        idx = idx / 2;
        int d = idx % 2;
        idx = idx / 2;
        int h = idx % 2;
        idx = idx / 2;
        int e = idx;

        find_10k_route(solIdx, f, d, h, e, q3, minQ1, maxQ1, minQ2, maxQ2);
    }
}

__device__ void try_pu_slide_angle_non_hau(struct PlatformSolution* sol, int solIdx, int angleIdx, int floorIdx, double s, float xVel1, float zVel1) {
    int angle = gArctanTableG[angleIdx];
    angle = (65536 + angle) % 65536;

    double t = tan(2.0 * M_PI * (double)(angle - (angle % 16)) / 65536.0);

    double n = (-(s * t) - 1.0 + sqrt((s * t - 1.0) * (s * t - 1.0) + 4.0 * s * s)) / (2.0 * s);
    float nTestX = gCosineTableG[angle >> 4] + n * gSineTableG[angle >> 4];

    if ((xVel1 < 0 && nTestX > 0) || (xVel1 > 0 && nTestX < 0)) {
        n = (-(s * t) - 1.0 - sqrt((s * t - 1.0) * (s * t - 1.0) + 4.0 * s * s)) / (2.0 * s);
    }

    double n1 = n / 0.05;

    if (fabs(n1) <= 1.01) {
        int minAngle = (int)round(65536.0 * asin(fabs(n1 / 1.01)) / (2 * M_PI));
        minAngle = minAngle + ((16 - (minAngle % 16)) % 16);

        for (int j = minAngle; j <= 32768 - minAngle; j += 16) {
            int j1 = (n1 >= 0) ? j : (65536 - j) % 65536;

            float idealMag = 32.0f * n1 / gSineTableG[j1 >> 4];

            float mag = find_closest_mag(idealMag);

            float vel0 = (float)(sqrt((double)xVel1 * (double)xVel1 + (double)zVel1 * (double)zVel1) / ((double)(mag / 32.0f) * (double)gCosineTableG[j1 >> 4] * 0.02 + 0.92));
            vel0 = (vel0 < 0) ? vel0 : -vel0;

            float xVel0 = vel0 * gSineTableG[angle >> 4];
            float zVel0 = vel0 * gCosineTableG[angle >> 4];

            float xVel1a = xVel0;
            float zVel1a = zVel0;

            float oldSpeed = sqrtf(xVel1a * xVel1a + zVel1a * zVel1a);

            xVel1a += zVel1a * (mag / 32.0f) * gSineTableG[j1 >> 4] * 0.05f;
            zVel1a -= xVel1a * (mag / 32.0f) * gSineTableG[j1 >> 4] * 0.05f;

            float newSpeed = sqrtf(xVel1a * xVel1a + zVel1a * zVel1a);

            xVel1a = xVel1a * oldSpeed / newSpeed;
            zVel1a = zVel1a * oldSpeed / newSpeed;

            xVel1a *= mag / 32.0f * gCosineTableG[j1 >> 4] * 0.02f + 0.92f;
            zVel1a *= mag / 32.0f * gCosineTableG[j1 >> 4] * 0.02f + 0.92f;

            float positionTest[3] = { sol->endPosition[0], sol->endPosition[1], sol->endPosition[2] };

            for (int s = 0; s < 4; s++) {
                positionTest[0] = positionTest[0] + sol->endTriangleNormals[floorIdx][1] * (xVel1a / 4.0f);
                positionTest[2] = positionTest[2] + sol->endTriangleNormals[floorIdx][1] * (zVel1a / 4.0f);
            }

            float floorHeight;
            int floorIdx1 = find_floor_triangles(positionTest, sol->endTriangles, sol->endTriangleNormals, &floorHeight);

            if (floorIdx1 != -1 && fabs(positionTest[1] - floorHeight) < 4.0f) {
                float prePositionTest[3] = { sol->penultimatePosition[0] + sol->penultimateFloorNormalY * xVel0 / 4.0f, sol->penultimatePosition[1], sol->penultimatePosition[2] + sol->penultimateFloorNormalY * zVel0 / 4.0f };

                if (!check_inbounds(prePositionTest)) {
                    int idx = atomicAdd(&nPUSolutions, 1);
                    if (idx < MAX_PU_SOLUTIONS) {
                        PUSolution solution;
                        solution.upwarpSolutionIdx = solIdx;
                        solution.returnSpeed = vel0;
                        solution.angle = angle;
                        solution.intendedDYaw = j1;
                        solution.stickMag = mag;
                        puSolutions[idx] = solution;
                    }
                }
            }
        }
    }
}

__device__ void find_pu_slide_setup(int solIdx) {
    struct UpwarpSolution* uwSol = &(upwarpSolutions[solIdx]);
    struct PlatformSolution* platSol = &(platSolutions[uwSol->platformSolutionIdx]);

    float floorHeight;
    int floorIdx = find_floor_triangles(platSol->endPosition, platSol->endTriangles, platSol->endTriangleNormals, &floorHeight);

    if (floorIdx != -1) {
        double s = (double)uwSol->pux / (double)uwSol->puz;

        float xVel1 = (float)(65536.0 * (double)uwSol->pux / (double)platSol->endTriangleNormals[floorIdx][1]);
        float zVel1 = (float)(65536.0 * (double)uwSol->puz / (double)platSol->endTriangleNormals[floorIdx][1]);

        for (int i = 0; i < 8192; i++) {
            try_pu_slide_angle_non_hau(platSol, solIdx, i, floorIdx, s, xVel1, zVel1);
        }
    }
}

__global__ void test_upwarp_solution() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nUpwarpSolutions, MAX_UPWARP_SOLUTIONS)) {
        find_pu_slide_setup(idx);
    }
}

void run_non_hau_bruteforcer(int g, int h, int i, int j, float normX, float normY, float normZ, float* host_norms, std::unordered_map<uint64_t, PUSolution>& puSolutionLookup, std::ofstream& wf, char* normalStages)
{
    cudaMemcpyToSymbol(nPUSolutions, &nPUSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

    test_upwarp_solution << < nBlocks, nThreads >> > ();

    cudaDeviceSynchronize();

    cudaMemcpyFromSymbol(&nPUSolutionsCPU, nPUSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);

    struct PUSolution* puSolutionsCPU = (struct PUSolution*)std::malloc(nPUSolutionsCPU * sizeof(struct PUSolution));

    if (nPUSolutionsCPU > 0) {
        normalStages[((g * nSamplesNY + h) * nSamplesNX + i) * nSamplesNZ + j] = 3;

        if (nPUSolutionsCPU > MAX_PU_SOLUTIONS) {
            fprintf(stderr, "Warning: Number of PU solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
            nPUSolutionsCPU = MAX_PU_SOLUTIONS;
        }
        puSolutionLookup.clear();

        // check_upwarp_solutions_for_the_right_one << < nBlocks, nThreads >> > ();

        cudaMemcpyFromSymbol(puSolutionsCPU, puSolutions, nPUSolutionsCPU * sizeof(struct PUSolution), 0, cudaMemcpyDeviceToHost);

        for (int l = 0; l < nPUSolutionsCPU; l++) {
            uint64_t key = (((uint64_t)puSolutionsCPU[l].upwarpSolutionIdx) << 32) | (reinterpret_cast<uint32_t&>(puSolutionsCPU[l].returnSpeed));
            puSolutionLookup[key] = puSolutionsCPU[l];
        }

        nPUSolutionsCPU = 0;

        for (std::pair<const uint64_t, PUSolution> p : puSolutionLookup) {
            puSolutionsCPU[nPUSolutionsCPU] = p.second;
            nPUSolutionsCPU++;
        }

        if (subSolutionPrintingMode == 1)
            printf("# PU Solutions (%d, %d, %d, %d): %d\n\n", g, h, i, j, nPUSolutionsCPU);
        else if (subSolutionPrintingMode == 2)
            printf("  Stage 3 Solutions: %d\n", nPlatSolutionsCPU);

        cudaMemcpyToSymbol(nPUSolutions, &nPUSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(puSolutions, puSolutionsCPU, nPUSolutionsCPU * sizeof(PUSolution), 0, cudaMemcpyHostToDevice);

        nBlocks = ((long long)16 * (long long)nPUSolutionsCPU + (long long)nThreads - (long long)1) / (long long)nThreads;

        check_pu_solutions_for_the_right_one << < nBlocks, nThreads >> > ();

        cudaMemcpyToSymbol(n10KSolutionsNonHAU, &n10KSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

        for (int q = minQ3; q <= maxQ3; q++) {
            test_pu_solution << <nBlocks, nThreads >> > (q, minQ1, maxQ1, 4 * (nPUFrames - 3) + minQ2, 4 * (nPUFrames - 3) + maxQ2);
        }

        cudaMemcpyFromSymbol(&n10KSolutionsCPU, n10KSolutionsNonHAU, sizeof(int), 0, cudaMemcpyDeviceToHost);
    }
    else
    {
        if (subSolutionPrintingMode == 2)
            printf("  Stage 3 Solutions: Failed\n");
    }

    if (n10KSolutionsCPU > 0) {
        normalStages[((g * nSamplesNY + h) * nSamplesNX + i) * nSamplesNZ + j] = 4;

        if (n10KSolutionsCPU > MAX_10K_SOLUTIONS_NON_HAU) {
            fprintf(stderr, "Warning: Number of 10K solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
            n10KSolutionsCPU = MAX_10K_SOLUTIONS_NON_HAU;

        }

        if (subSolutionPrintingMode == 1)
            printf("# 10k Solutions (%d, %d, %d, %d): %d\n\n", g, h, i, j, n10KSolutionsCPU);
        else if (subSolutionPrintingMode == 2)
            printf("  Stage 4 Solutions: %d\n", n10KSolutionsCPU);

        struct PlatformSolution* platSolutionsCPU = (struct PlatformSolution*)std::malloc(nPlatSolutionsCPU * sizeof(struct PlatformSolution));
        struct UpwarpSolution* upwarpSolutionsCPU = (struct UpwarpSolution*)std::malloc(nUpwarpSolutionsCPU * sizeof(struct UpwarpSolution));
        struct TenKSolutionNonHAU* tenKSolutionsCPU = (struct TenKSolutionNonHAU*)std::malloc(n10KSolutionsCPU * sizeof(struct TenKSolutionNonHAU));

        cudaMemcpyFromSymbol(tenKSolutionsCPU, tenKSolutionsNonHAU, n10KSolutionsCPU * sizeof(struct TenKSolutionNonHAU), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(upwarpSolutionsCPU, upwarpSolutions, nUpwarpSolutionsCPU * sizeof(struct UpwarpSolution), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(platSolutionsCPU, platSolutions, nPlatSolutionsCPU * sizeof(struct PlatformSolution), 0, cudaMemcpyDeviceToHost);

        for (int l = 0; l < n10KSolutionsCPU; l++) {
            struct TenKSolutionNonHAU* tenKSol = &(tenKSolutionsCPU[l]);
            struct PUSolution* puSol = &(puSolutionsCPU[tenKSol->puSolutionIdx]);
            struct UpwarpSolution* uwSol = &(upwarpSolutionsCPU[puSol->upwarpSolutionIdx]);
            struct PlatformSolution* platSol = &(platSolutionsCPU[uwSol->platformSolutionIdx]);

            wf << normX << ", " << normY << ", " << normZ << ", ";
            wf << tenKSol->startPosition[0] << ", " << tenKSol->startPosition[1] << ", " << tenKSol->startPosition[2] << ", ";
            wf << tenKSol->frame1Position[0] << ", " << tenKSol->frame1Position[1] << ", " << tenKSol->frame1Position[2] << ", ";
            wf << tenKSol->frame2Position[0] << ", " << tenKSol->frame2Position[1] << ", " << tenKSol->frame2Position[2] << ", ";
            wf << platSol->returnPosition[0] << ", " << platSol->returnPosition[1] << ", " << platSol->returnPosition[2] << ", ";
            wf << tenKSol->pre10Kspeed << ", " << tenKSol->pre10KVel[0] << ", " << tenKSol->pre10KVel[1] << ", ";
            wf << puSol->returnSpeed << ", " << tenKSol->returnVel[0] << ", " << tenKSol->returnVel[1] << ", ";
            wf << tenKSol->frame1QSteps << ", " << tenKSol->frame2QSteps << ", " << tenKSol->frame3QSteps << ", ";
            wf << tenKSol->stick10K[0] << ", " << tenKSol->stick10K[1] << ", ";
            wf << tenKSol->cameraYaw10K << ", ";
            wf << host_norms[3 * tenKSol->startFloorIdx] << ", " << host_norms[3 * tenKSol->startFloorIdx + 1] << ", " << host_norms[3 * tenKSol->startFloorIdx + 2] << ", ";
            wf << tenKSol->startPositionLimits[0][0] << ", " << tenKSol->startPositionLimits[0][1] << ", " << tenKSol->startPositionLimits[0][2] << ", ";
            wf << tenKSol->startPositionLimits[1][0] << ", " << tenKSol->startPositionLimits[1][1] << ", " << tenKSol->startPositionLimits[1][2] << ", ";
            wf << platSol->nFrames << ", ";
            wf << platSol->endNormal[0] << ", " << platSol->endNormal[1] << ", " << platSol->endNormal[2] << ", ";
            wf << platSol->endPosition[0] << ", " << platSol->endPosition[1] << ", " << platSol->endPosition[2] << ", ";
            wf << uwSol->upwarpPosition[0] << ", " << uwSol->upwarpPosition[1] << ", " << uwSol->upwarpPosition[2] << ", ";
            wf << uwSol->pux << ", " << uwSol->puz << ", ";
            wf << puSol->angle << ", " << puSol->stickMag << ", " << puSol->intendedDYaw << std::endl;
        }

        free(tenKSolutionsCPU);
        free(upwarpSolutionsCPU);
        free(platSolutionsCPU);
    }
    else
    {
        if (subSolutionPrintingMode == 2)
            printf("  Stage 4 Solutions: Failed\n");
    }

    free(puSolutionsCPU);
}


// Checks to see if a PU solution with the correct parameters was found.
__global__ void check_pu_solutions_for_the_right_one()
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < min(nPUSolutions, MAX_PU_SOLUTIONS)) {
        struct PUSolution* puSol = &(puSolutions[idx]);

        if (puSol->angle == 63261 && abs(puSol->stickMag - 8.632811546) < 0.0001 && puSol->intendedDYaw == 36256)
        {
            struct UpwarpSolution* upSol = &(upwarpSolutions[puSol->upwarpSolutionIdx]);

            if (upSol->pux == 752 && upSol->puz == -3304)
            {
                struct PlatformSolution* platSol = &(platSolutions[upSol->platformSolutionIdx]);

                printf("Matching Upwarp Solution Found!\n Index: %i\n PlatSolEndNormal: %f, %f, %f\n\n", idx, platSol->endNormal[0], platSol->endNormal[1], platSol->endNormal[2]);

                if (abs(platSol->endNormal[0] - 0.2320000678) < 0.0001 && abs(platSol->endNormal[1] - 0.8384857774) < 0.0001 && abs(platSol->endNormal[2] - 0.4581809938) < 0.0001)
                {
                    printf("Correct PU Solution found! Index: %i\n", idx);

                    correctPuSolIdx = idx;
                }
            }

            printf("\n");
        }
    }
}

void setup_output_non_hau(std::ofstream& wf)
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
    wf << "10K Stick X, 10K Stick Y, ";
    wf << "10K Camera Yaw, ";
    wf << "Start Floor Normal X, Start Floor Normal Y, Start Floor Normal Z, ";
    wf << "Start Position Limit 1 X, Start Position Limit 1 Y, Start Position Limit 1 Z, ";
    wf << "Start Position Limit 2 X, Start Position Limit 2 Y, Start Position Limit 2 Z, ";
    wf << "Number of Tilt Frames, ";
    wf << "Post-Tilt Platform Normal X, Post-Tilt Platform Normal Y, Post-Tilt Platform Normal Z, ";
    wf << "Post-Tilt Position X, Post-Tilt Position Y, Post-Tilt Position Z, ";
    wf << "Post-Upwarp Position X, Post-Upwarp Position Y, Post-Upwarp Position Z, ";
    wf << "Upwarp PU X, Upwarp PU Z, ";
    wf << "Upwarp Slide Facing Angle, Upwarp Slide IntendedMag, Upwarp Slide IntendedDYaw" << std::endl;
}
