#include <fstream>
#include <cstring>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <unordered_map>
#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include <math.h>
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#include "Platform.hpp"
#include "vmath.hpp"

#include "BruteforceFunctions.cuh"
#include "BruteforceVariables.cuh"
#include "Floors.cuh"
#include "SolutionStructs.cuh"


std::ofstream out_stream;

int main(int argc, char* argv[]) {
    int nThreads = 256;
    size_t memorySize = 10000000;

    int minQ1 = 1;
    int maxQ1 = 4;
    int minQ2 = 1;
    int maxQ2 = 4;
    int minQ3 = 1;
    int maxQ3 = 4;

    int nPUFrames = 3;
    int maxFrames = 200;

    float minNX = -0.3958f;
    float maxNX = -0.3958f;
    float minNZ = 0.307f;
    float maxNZ = 0.307f;
    float minNY = 0.87f;
    float maxNY = 0.87f;

    int nSamplesNX = 1;
    int nSamplesNZ = 1;
    int nSamplesNY = 1;

    nSamplesNX = minNX == maxNX ? 1 : nSamplesNX;
    nSamplesNY = minNY == maxNY ? 1 : nSamplesNY;
    nSamplesNZ = minNZ == maxNZ ? 1 : nSamplesNZ;

    float deltaX = 0.5f;
    float deltaZ = 0.5f;

    Vec3f platformPos = {-1945.0f, -3225.0f, -715.0f};

    std::string outFile = "outData.csv";

    bool verbose = false;
    
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("BitFS Platform Max Tilt Brute Forcer.\n");
            printf("This program accepts the following options:\n\n");
            printf("-f <frames>: Maximum frames of platform tilt considered.\n");
            printf("             Default: %d\n", maxFrames);
            printf("-p <frames>: Number of frames of PU movement for 10k glitch\n");
            printf("             Default: %d\n", nPUFrames);
            printf("-q1 <min_q1> <max_q1>: Range of q-frames to test for frame 1 of 10k PU route.\n");
            printf("                       Default: %d %d\n", minQ1, maxQ1);
            printf("-q2 <min_q2> <max_q2>: Range of q-frames to test for frame 2 of 10k PU route.\n");
            printf("                       Default: %d %d\n", minQ2, maxQ2);
            printf("-q3 <min_q3> <max_q3>: Range of q-frames to test for frame 3 of 10k PU route.\n");
            printf("                       Default: %d %d\n", minQ3, maxQ3);
            printf("-nx <min_nx> <max_nx> <n_samples>: Inclusive range of x normals to be considered, and the number of normals to sample.\n");
            printf("                                   If min_nx==max_nx then n_samples will be set to 1.\n");
            printf("                                   Default: %g %g %d\n", minNX, maxNX, nSamplesNX);
            printf("-nz <min_nz> <max_nz> <n_samples>: Inclusive range of z normals to be considered, and the number of normals to sample.\n");
            printf("                                   If min_nz==max_nz then n_samples will be set to 1.\n");
            printf("                                   Default: %g %g %d\n", minNZ, maxNZ, nSamplesNZ);
            printf("-ny <min_ny> <max_ny> <n_samples>: Inclusive range of y normals to be considered, and the number of normals to sample.\n");
            printf("                                   If min_ny==max_ny then n_samples will be set to 1.\n");
            printf("                                   Default: %g %g %d\n", minNY, maxNY, nSamplesNY);
            printf("-dx <delta_x>: x coordinate spacing of positions on the platform.\n");
            printf("               Default: %g\n", deltaX);
            printf("-dz <delta_z>: z coordinate spacing of positions on the platform.\n");
            printf("               Default: %g\n", deltaZ);
            printf("-p <platform_x> <platform_y> <platform_z>: Position of the pyramid platform.\n");
            printf("                                           Default: %g %g %g\n", platformPos[0], platformPos[1], platformPos[2]);
            printf("-o: Path to the output file.\n");
            printf("    Default: %s\n", outFile.c_str());
            printf("-t <threads>: Number of CUDA threads to assign to the program.\n");
            printf("              Default: %d\n", nThreads);
            printf("-m <memory>: Amount of GPU memory to assign to the program.\n");
            printf("             Default: %d\n", memorySize);
            printf("-v: Verbose mode. Prints all parameters used in brute force.\n");
            printf("    Default: off\n");
            printf("-h --help: Prints this text.\n");
            exit(0);
        }
        else if (!strcmp(argv[i], "-f")) {
            maxFrames = std::stoi(argv[i + 1]);

            i += 1;
        }
        else if (!strcmp(argv[i], "-q1")) {
            minQ1 = std::stoi(argv[i + 1]);
            maxQ1 = std::stoi(argv[i + 2]);

            i += 2;
        }
        else if (!strcmp(argv[i], "-q2")) {
            minQ2 = std::stoi(argv[i + 1]);
            maxQ2 = std::stoi(argv[i + 2]);

            i += 2;
        }
        else if (!strcmp(argv[i], "-q3")) {
            minQ3 = std::stoi(argv[i + 1]);
            maxQ3 = std::stoi(argv[i + 2]);

            i += 2;
        }
        else if (!strcmp(argv[i], "-p")) {
            nPUFrames = std::stoi(argv[i + 1]);

            i += 1;
        }
        else if (!strcmp(argv[i], "-t")) {
            nThreads = std::stoi(argv[i + 1]);

            i += 1;
        }
        else if (!strcmp(argv[i], "-m")) {
            memorySize = std::stoi(argv[i + 1]);

            i += 1;
        }
        else if (!strcmp(argv[i], "-nx")) {
            minNX = std::stof(argv[i + 1]);
            maxNX = std::stof(argv[i + 2]);

            if (minNX == maxNX) {
                nSamplesNX = 1;
            }
            else {
                nSamplesNX = std::stoi(argv[i + 3]);
            }

            i += 3;
        }
        else if (!strcmp(argv[i], "-nz")) {
            minNZ = std::stof(argv[i + 1]);
            maxNZ = std::stof(argv[i + 2]);

            if (minNZ == maxNZ) {
                nSamplesNZ = 1;
            }
            else {
                nSamplesNZ = std::stoi(argv[i + 3]);
            }

            i += 3;
        }
        else if (!strcmp(argv[i], "-ny")) {
            minNY = std::stof(argv[i + 1]);
            maxNY = std::stof(argv[i + 2]);

            if (minNY == maxNY) {
                nSamplesNY = 1;
            }
            else {
                nSamplesNY = std::stoi(argv[i + 3]);
            }

            i += 3;
        }
        else if (!strcmp(argv[i], "-dx")) {
            deltaX = std::stof(argv[i + 1]);
            i += 1;
        }
        else if (!strcmp(argv[i], "-dz")) {
            deltaZ = std::stof(argv[i + 1]);
            i += 1;
        }
        else if (!strcmp(argv[i], "-p")) {
            platformPos[0] = std::stof(argv[i + 1]);
            platformPos[1] = std::stof(argv[i + 2]);
            platformPos[2] = std::stof(argv[i + 3]);
            i += 3;
        }
        else if (!strcmp(argv[i], "-o")) {
            outFile = argv[i + 1];
            i += 1;
        }
        else if (!strcmp(argv[i], "-v")) {
            verbose = true;
        }
    }
    
    if (nPUFrames != 3) {
        fprintf(stderr, "Error: This brute forcer currently only supports 3 frame 10k routes. Value selected: %d.", nPUFrames);
        return 1;
    }

    if (verbose) {
        printf("Max Frames: %d\n", maxFrames);
        printf("10K Frame 1 Q-Frames: (%d, %d)\n", minQ1, maxQ1);
        printf("10K Frame 2 Q-Frames: (%d, %d)\n", minQ2, maxQ2);
        printf("10K Frame 3 Q-Frames: (%d, %d)\n", minQ3, maxQ3);
        printf("Off Platform Frames: %d\n", nPUFrames);
        printf("Off Platform Frames: %d\n", nPUFrames);
        printf("Off Platform Frames: %d\n", nPUFrames);
        printf("X Normal Range: (%g, %g)\n", minNX, maxNX);
        printf("Z Normal Range: (%g, %g)\n", minNZ, maxNZ);
        printf("Y Normal Range: (%g, %g)\n", minNY, maxNY);
        printf("X Normal Samples: %d\n", nSamplesNX);
        printf("Z Normal Samples: %d\n", nSamplesNZ);
        printf("Y Normal Samples: %d\n", nSamplesNY);
        printf("X Spacing: %g\n", deltaX);
        printf("Z Spacing: %g\n", deltaZ);
        printf("Platform Position: (%g, %g, %g)\n", platformPos[0], platformPos[1], platformPos[2]);
        printf("\n");
    }

    init_reverse_atan<< <1, 1>> >();
    init_mag_set<< <1, 1>> >();
    initialise_floors<< <1, 1>> >();
    set_platform_pos<< <1, 1>> >(platformPos[0], platformPos[1], platformPos[2]);

    short* dev_tris;
    float* dev_norms;
    short* host_tris = (short*)std::malloc(18 * sizeof(short));
    float* host_norms = (float*)std::malloc(6 * sizeof(float));

    cudaMalloc((void**)&dev_tris, 18*sizeof(short));
    cudaMalloc((void**)&dev_norms, 6*sizeof(float));

    const float deltaNX = (nSamplesNX > 1) ? (maxNX - minNX) / (nSamplesNX - 1) : 0;
    const float deltaNZ = (nSamplesNZ > 1) ? (maxNZ - minNZ) / (nSamplesNZ - 1) : 0;
    const float deltaNY = (nSamplesNY > 1) ? (maxNY - minNY) / (nSamplesNY - 1) : 0;

    std::ofstream wf(outFile);
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
    wf << "Upwarp PU X, Upwarp PU Z" << std::endl;

    for (int h = 0; h < nSamplesNY; h++) {
        for (int i = 0; i < nSamplesNX; i++) {
            for (int j = 0; j < nSamplesNZ; j++) {
                float normX = minNX + i * deltaNX;
                float normZ = minNZ + j * deltaNZ;
                float normY = minNY + h * deltaNY;

                Vec3f startNormal = { normX, normY, normZ };
                Platform platform = Platform(platformPos[0], platformPos[1], platformPos[2], startNormal);

                bool squishTest = (platform.ceilings[0].normal[1] > -0.5f) || (platform.ceilings[1].normal[1] > -0.5f) || (platform.ceilings[2].normal[1] > -0.5f) || (platform.ceilings[3].normal[1] > -0.5f);

                if (squishTest) {
                    set_squish_ceilings << <1, 1 >> > (platform.ceilings[0].normal[1], platform.ceilings[1].normal[1], platform.ceilings[2].normal[1], platform.ceilings[3].normal[1]);
                    Vec3f position = { 0.0f, 0.0f, 0.0f };

                    for (int k = 0; k < nPUFrames; k++) {
                        platform.platform_logic(position);

                        if (k == 0) {
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

                            set_start_triangle<< <1, 1>> >(dev_tris, dev_norms);
                        }
                    }

                    int nBlocks = (112 + nThreads - 1) / nThreads;

                    calculate_10k_multipliers << <nBlocks, nThreads >> > (minQ1 + minQ2, maxQ1 + maxQ2, minQ3, maxQ3);

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

                    if (nX * nZ > memorySize) {
                        printf("Warning: GPU buffer too small for normal (%g, %g), skipping.\n", normX, normZ);
                        continue;
                    }

                    int nPlatSolutionsCPU = 0;
                    int nUpwarpSolutionsCPU = 0;
                    int nStickSolutionsCPU = 0;
                    int nOUPSolutionsCPU = 0;
                    int nSpeedSolutionsCPU = 0;
                    int n10KSolutionsCPU = 0;

                    cudaMemcpyToSymbol(nPlatSolutions, &nPlatSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

                    nBlocks = (nX * nZ + nThreads - 1) / nThreads;

                    cudaFunc<< <nBlocks, nThreads>> >(minX, deltaX, minZ, deltaZ, nX, nZ, platform.normal[0], platform.normal[1], platform.normal[2], maxFrames);

                    cudaMemcpyFromSymbol(&nPlatSolutionsCPU, nPlatSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);

                    if (nPlatSolutionsCPU > MAX_PLAT_SOLUTIONS) {
                        fprintf(stderr, "Warning: Number of platform solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                        nPlatSolutionsCPU = MAX_PLAT_SOLUTIONS;
                    }

                    if (nPlatSolutionsCPU > 0) {
                        printf("---------------------------------------\nTesting Normal: %g, %g, %g\n        Index: %d, %d, %d\n", normX, normY, normZ, h, i, j);
                        printf("        # Platform Solutions: %d\n", nPlatSolutionsCPU);

                        cudaMemcpyToSymbol(nUpwarpSolutions, &nUpwarpSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

                        nBlocks = (nPlatSolutionsCPU + nThreads - 1) / nThreads;

                        find_upwarp_solutions<< <nBlocks, nThreads>> >();

                        cudaMemcpyFromSymbol(&nUpwarpSolutionsCPU, nUpwarpSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
                    }

                    if (nUpwarpSolutionsCPU > 0) {
                        if (nUpwarpSolutionsCPU > MAX_UPWARP_SOLUTIONS) {
                            fprintf(stderr, "Warning: Number of upwarp solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                            nUpwarpSolutionsCPU = MAX_UPWARP_SOLUTIONS;
                        }

                        printf("        # Upwarp Solutions: %d\n", nUpwarpSolutionsCPU);

                        cudaMemcpyToSymbol(nStickSolutions, &nStickSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

                        nBlocks = (nUpwarpSolutionsCPU + nThreads - 1) / nThreads;

                        find_stick_solutions<< <nBlocks, nThreads>> >();

                        cudaMemcpyFromSymbol(&nStickSolutionsCPU, nStickSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
                    }

                    if (nStickSolutionsCPU > 0) {
                        if (nStickSolutionsCPU > MAX_STICK_SOLUTIONS) {
                            fprintf(stderr, "Warning: Number of stick solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                            nStickSolutionsCPU = MAX_STICK_SOLUTIONS;
                        }

                        printf("        # Stick Solutions: %d\n", nStickSolutionsCPU);

                        cudaMemcpyToSymbol(nOUPSolutions, &nOUPSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

                        nBlocks = (2048 * nStickSolutionsCPU + nThreads - 1) / nThreads;

                        check_speed_angle<< <nBlocks, nThreads>> >();

                        cudaMemcpyFromSymbol(&nOUPSolutionsCPU, nOUPSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
                    }

                    if (nOUPSolutionsCPU > 0) {
                        if (nOUPSolutionsCPU > MAX_OUP_SOLUTIONS) {
                            fprintf(stderr, "Warning: Number of 1-up Platform solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                            nOUPSolutionsCPU = MAX_OUP_SOLUTIONS;
                        }

                        printf("        # 1-up Platform Solutions: %d\n", nOUPSolutionsCPU);

                        cudaMemcpyToSymbol(nSpeedSolutions, &nSpeedSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

                        nBlocks = (nOUPSolutionsCPU + nThreads - 1) / nThreads;

                        test_oup_solution<< <nBlocks, nThreads>> >();

                        cudaMemcpyFromSymbol(&nSpeedSolutionsCPU, nSpeedSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
                    }

                    if (nSpeedSolutionsCPU > 0) {
                        if (nSpeedSolutionsCPU > MAX_SPEED_SOLUTIONS) {
                            fprintf(stderr, "Warning: Number of speed solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                            nSpeedSolutionsCPU = MAX_SPEED_SOLUTIONS;

                        }

                        printf("        # Speed Solutions: %d\n", nSpeedSolutionsCPU);

                        cudaMemcpyToSymbol(n10KSolutionsHAU, &n10KSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

                        nBlocks = (nSpeedSolutionsCPU + nThreads - 1) / nThreads;

                        test_speed_solution<< <nBlocks, nThreads>> >();

                        cudaMemcpyFromSymbol(&n10KSolutionsCPU, n10KSolutionsHAU, sizeof(int), 0, cudaMemcpyDeviceToHost);
                    }

                    if (n10KSolutionsCPU > 0) {
                        if (n10KSolutionsCPU > MAX_10K_SOLUTIONS_HAU) {
                            fprintf(stderr, "Warning: Number of 10k solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                            n10KSolutionsCPU = MAX_10K_SOLUTIONS_HAU;
                        }

                        printf("        # 10k Solutions: %d\n", n10KSolutionsCPU);

                        struct PlatformSolution* platSolutionsCPU = (struct PlatformSolution*)std::malloc(nPlatSolutionsCPU * sizeof(struct PlatformSolution));
                        struct UpwarpSolution* upwarpSolutionsCPU = (struct UpwarpSolution*)std::malloc(nUpwarpSolutionsCPU * sizeof(struct UpwarpSolution));
                        struct StickSolution* stickSolutionsCPU = (struct StickSolution*)std::malloc(nStickSolutionsCPU * sizeof(struct StickSolution));
                        struct OUPSolution* oupSolutionsCPU = (struct OUPSolution*)std::malloc(nOUPSolutionsCPU * sizeof(struct OUPSolution));
                        struct SpeedSolution* speedSolutionsCPU = (struct SpeedSolution*)std::malloc(nSpeedSolutionsCPU * sizeof(struct SpeedSolution));
                        struct TenKSolutionHAU* tenKSolutionsCPU = (struct TenKSolutionHAU*)std::malloc(n10KSolutionsCPU * sizeof(struct TenKSolutionHAU));

                        cudaMemcpyFromSymbol(tenKSolutionsCPU, tenKSolutionsHAU, n10KSolutionsCPU * sizeof(struct TenKSolutionHAU), 0, cudaMemcpyDeviceToHost);
                        cudaMemcpyFromSymbol(speedSolutionsCPU, speedSolutions, nSpeedSolutionsCPU * sizeof(struct SpeedSolution), 0, cudaMemcpyDeviceToHost);
                        cudaMemcpyFromSymbol(oupSolutionsCPU, oupSolutions, nOUPSolutionsCPU * sizeof(struct OUPSolution), 0, cudaMemcpyDeviceToHost);
                        cudaMemcpyFromSymbol(stickSolutionsCPU, stickSolutions, nStickSolutionsCPU * sizeof(struct StickSolution), 0, cudaMemcpyDeviceToHost);
                        cudaMemcpyFromSymbol(upwarpSolutionsCPU, upwarpSolutions, nUpwarpSolutionsCPU * sizeof(struct UpwarpSolution), 0, cudaMemcpyDeviceToHost);
                        cudaMemcpyFromSymbol(platSolutionsCPU, platSolutions, nPlatSolutionsCPU * sizeof(struct PlatformSolution), 0, cudaMemcpyDeviceToHost);

                        for (int l = 0; l < n10KSolutionsCPU; l++) {
                            struct TenKSolutionHAU* tenKSol = &(tenKSolutionsCPU[l]);
                            struct SpeedSolution* speedSol = &(speedSolutionsCPU[tenKSol->speedSolutionIdx]);
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
                }
            }
        }
    }

    free(host_tris);
    free(host_norms);
    cudaFree(dev_tris);
    cudaFree(dev_norms);
    wf.close();
}