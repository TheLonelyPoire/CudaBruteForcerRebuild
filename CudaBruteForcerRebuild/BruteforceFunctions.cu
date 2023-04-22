#include "BruteforceFunctions.cuh"

#include "math.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#include "Platform.cuh"
#include "vmath.hpp"

#include "BruteforceVariables.cuh"
#include "CommonFunctions.cuh"
#include "Floors.cuh"
#include "HAUFunctions.cuh"
#include "NonHAUFunctions.cuh"
#include "PlatformSolutionFunctions.cuh"
#include "UpwarpSolutionFunctions.cuh"
#include "RunParameters.hpp"


// Common Bruteforce Functions

void run_common_bruteforcer(int g, int h, int i, int j, float normX, float normY, float normZ, short* host_tris, float* host_norms, short* dev_tris, float* dev_norms, std::unordered_map<uint64_t, PUSolution>& puSolutionLookup, std::ofstream& wf, char* normalStages, float* finalHeightDiffs)
{
    Vec3f startNormal = { normX, normY, normZ };
    Platform platform = Platform(platformPos[0], platformPos[1], platformPos[2], startNormal);

    bool squishTest = (platform.ceilings[0].normal[1] > -0.5f) || (platform.ceilings[1].normal[1] > -0.5f) || (platform.ceilings[2].normal[1] > -0.5f) || (platform.ceilings[3].normal[1] > -0.5f);

    if (squishTest) {
        set_squish_ceilings << < 1, 1 >> > (platform.ceilings[0].normal[1], platform.ceilings[1].normal[1], platform.ceilings[2].normal[1], platform.ceilings[3].normal[1]);
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

                set_start_triangle << < 1, 1 >> > (dev_tris, dev_norms);
            }
        }

        if (solverMode)
        {
            nBlocks = (112 + nThreads - 1) / nThreads;
            calculate_10k_multipliers << <nBlocks, nThreads >> > (minQ1 + minQ2, maxQ1 + maxQ2, minQ3, maxQ3);
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

        if (nX * nZ > memorySize) {
            printf("Warning: GPU buffer too small for normal (%g, %g), skipping.\n", normX, normZ);
            return;
        }

        // Common

        nPlatSolutionsCPU = 0;
        nUpwarpSolutionsCPU = 0;
        n10KSolutionsCPU = 0;

        // Non-HAU Aligned

        nPUSolutionsCPU = 0;

        // HAU-Aligned

        nStickSolutionsCPU = 0;
        nOUPSolutionsCPU = 0;
        nSpeedSolutionsCPU = 0;

        if(subSolutionPrintingMode == 2)
            printf("---------------------------------------\nTesting Normal: %g, %g, %g\n  Index: %d, %d, %d, %d\n", normX, normY, normZ, g, h, i, j);

        cudaMemcpyToSymbol(nPlatSolutions, &nPlatSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

        nBlocks = (nX * nZ + nThreads - 1) / nThreads;

        cudaFunc << < nBlocks, nThreads >> > (minX, deltaX, minZ, deltaZ, nX, nZ, platform.normal[0], platform.normal[1], platform.normal[2], maxFrames);

        cudaMemcpyFromSymbol(&nPlatSolutionsCPU, nPlatSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);

        if (nPlatSolutionsCPU > 0) {
            normalStages[((g * nSamplesNY + h) * nSamplesNX + i) * nSamplesNZ + j] = 1;

            if (nPlatSolutionsCPU > MAX_PLAT_SOLUTIONS) {
                fprintf(stderr, "Warning: Number of platform solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                nPlatSolutionsCPU = MAX_PLAT_SOLUTIONS;
            }

            if(subSolutionPrintingMode == 2)
                printf("  Stage 1 Solutions: %d\n", nPlatSolutionsCPU);

            nBlocks = (nPlatSolutionsCPU + nThreads - 1) / nThreads;

            //check_platform_solutions_for_the_right_one << < nBlocks, nThreads >> > ();

            cudaMemcpyToSymbol(nUpwarpSolutions, &nUpwarpSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

            find_upwarp_solutions << < nBlocks, nThreads >> > (1000000000.0f);

            cudaDeviceSynchronize();

            cudaMemcpyFromSymbol(&nUpwarpSolutionsCPU, nUpwarpSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);


        }
        else
        {
            if(subSolutionPrintingMode == 2)
                printf("  Stage 1 Solutions: Failed\n");
        }

        if (nUpwarpSolutionsCPU > 0) {
            normalStages[((g * nSamplesNY + h) * nSamplesNX + i) * nSamplesNZ + j] = 2;

            if (nUpwarpSolutionsCPU > MAX_UPWARP_SOLUTIONS) {
                fprintf(stderr, "Warning: Number of upwarp solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                nUpwarpSolutionsCPU = MAX_UPWARP_SOLUTIONS;
            }

            if(subSolutionPrintingMode == 2)
                printf("  Stage 2 Solutions: %d\n", nUpwarpSolutionsCPU);

            if (!stopAtUpwarp)
            {
                nBlocks = (nUpwarpSolutionsCPU + nThreads - 1) / nThreads;

                switch(solverMode)
                {
                case 0:
                    run_non_hau_bruteforcer(g, h, i, j, normX, normY, normZ, host_norms, puSolutionLookup, wf, normalStages);
                    break;
                case 1:
                    run_hau_bruteforcer(g, h, i, j, normX, normY, normZ, host_norms, wf, normalStages, finalHeightDiffs);
                    break;
                case 2:
                    std::cout << "IMPLEMENT SLIDE KICK HERE\n";
                    break;
                }
            }
        }
        else
        {
            if (subSolutionPrintingMode == 2)
            {
                // TODO: Update for more than two solvers
                printf("  Stage 2 Solutions: Failed\n");
                printf("  Stage 3 Solutions: Failed\n");
                printf("  Stage 4 Solutions: Failed\n");
                if (solverMode)
                {
                    printf("  Stage 5 Solutions: Failed\n");
                    printf("  Stage 6 Pass Count: Failed\n");
                    printf("  Stage 7 Pass Count: Failed\n");
                    printf("  Stage 8 Pass Count: Failed\n");
                    printf("  Stage 9 Solutions: Failed\n");
                }
            }
        }
    }
}

// Computes the height wiggle room number for the specified normal.
void run_max_elevation_computations(int g, int h, int i, int j, float normX, float normY, float normZ, short* host_tris, float* host_norms, short* dev_tris, float* dev_norms, std::unordered_map<uint64_t, PUSolution>& puSolutionLookup, std::ofstream& wf, float* platformHWRs)
{
    Vec3f startNormal = { normX, normY, normZ };
    Platform platform = Platform(platformPos[0], platformPos[1], platformPos[2], startNormal);

    bool squishTest = (platform.ceilings[0].normal[1] > -0.5f) || (platform.ceilings[1].normal[1] > -0.5f) || (platform.ceilings[2].normal[1] > -0.5f) || (platform.ceilings[3].normal[1] > -0.5f);

    if (squishTest) {
        set_squish_ceilings << < 1, 1 >> > (platform.ceilings[0].normal[1], platform.ceilings[1].normal[1], platform.ceilings[2].normal[1], platform.ceilings[3].normal[1]);
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

                set_start_triangle << < 1, 1 >> > (dev_tris, dev_norms);
            }
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

        if (nX * nZ > memorySize) {
            printf("Warning: GPU buffer too small for normal (%g, %g), skipping.\n", normX, normZ);
            return;
        }

        // Common

        nPlatSolutionsCPU = 0;
        nUpwarpSolutionsCPU = 0;
        n10KSolutionsCPU = 0;

        // Non-HAU Aligned

        nPUSolutionsCPU = 0;

        // HAU-Aligned

        nStickSolutionsCPU = 0;
        nOUPSolutionsCPU = 0;
        nSpeedSolutionsCPU = 0;

        cudaMemcpyToSymbol(nPlatSolutions, &nPlatSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

        nBlocks = (nX * nZ + nThreads - 1) / nThreads;

        //std::cout << "Platform Normal: " << platform.normal[0] << ", " << platform.normal[1] << "," << platform.normal[2] << "\n";

        cudaFunc << < nBlocks, nThreads >> > (minX, deltaX, minZ, deltaZ, nX, nZ, platform.normal[0], platform.normal[1], platform.normal[2], maxFrames);

        cudaMemcpyFromSymbol(&nPlatSolutionsCPU, nPlatSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);

        if (nPlatSolutionsCPU > 0) {
            printf("---------------------------------------\nTesting Normal: %g, %g, %g\n        Index: %d, %d, %d, %d\n", normX, normY, normZ, g, h, i, j);
            printf("        # Platform Solutions: %d\n", nPlatSolutionsCPU);

            if (nPlatSolutionsCPU > MAX_PLAT_SOLUTIONS) {
                fprintf(stderr, "Warning: Number of platform solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                nPlatSolutionsCPU = MAX_PLAT_SOLUTIONS;
            }

            struct PlatformSolution* platSolutionsCPU = (struct PlatformSolution*)std::malloc(nPlatSolutionsCPU * sizeof(struct PlatformSolution));

            cudaMemcpyFromSymbol(platSolutionsCPU, platSolutions, nPlatSolutionsCPU * sizeof(struct PlatformSolution), 0, cudaMemcpyDeviceToHost);

            cudaDeviceSynchronize();

            float maxHeight = -3000;

            for (int i = 0; i < nPlatSolutionsCPU; ++i)
            {
                if ((&platSolutionsCPU[i])->returnPosition[1] > maxHeight)
                {
                    maxHeight = (&platSolutionsCPU[i])->returnPosition[1];
                }
            }

            platformHWRs[((g * nSamplesNY + h) * nSamplesNX + i) * nSamplesNZ + j] = maxHeight + 2971;
            std::cout << "HWR: " << maxHeight + 2971 << "\n\n";

            free(platSolutionsCPU);

        }
        else
        {
            printf("No platform solutions found for normal: %f, %f, %f.\n", normX, normY, normZ);
        }
    }
}

// Computes the minimum upwarp speed for the specified normal.
void run_min_upwarp_speed_computations(int g, int h, int i, int j, float normX, float normY, float normZ, short* host_tris, float* host_norms, short* dev_tris, float* dev_norms, std::unordered_map<uint64_t, PUSolution>& puSolutionLookup, std::ofstream& wf, float* minUpwarpSpeeds)
{
    Vec3f startNormal = { normX, normY, normZ };
    Platform platform = Platform(platformPos[0], platformPos[1], platformPos[2], startNormal);

    bool squishTest = (platform.ceilings[0].normal[1] > -0.5f) || (platform.ceilings[1].normal[1] > -0.5f) || (platform.ceilings[2].normal[1] > -0.5f) || (platform.ceilings[3].normal[1] > -0.5f);

    if (squishTest) {
        set_squish_ceilings << < 1, 1 >> > (platform.ceilings[0].normal[1], platform.ceilings[1].normal[1], platform.ceilings[2].normal[1], platform.ceilings[3].normal[1]);
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

                set_start_triangle << < 1, 1 >> > (dev_tris, dev_norms);
            }
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

        if (nX * nZ > memorySize) {
            printf("Warning: GPU buffer too small for normal (%g, %g), skipping.\n", normX, normZ);
            return;
        }

        // Common

        nPlatSolutionsCPU = 0;
        nUpwarpSolutionsCPU = 0;
        n10KSolutionsCPU = 0;

        // Non-HAU Aligned

        nPUSolutionsCPU = 0;

        // HAU-Aligned

        nStickSolutionsCPU = 0;
        nOUPSolutionsCPU = 0;
        nSpeedSolutionsCPU = 0;

        cudaMemcpyToSymbol(nPlatSolutions, &nPlatSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

        nBlocks = (nX * nZ + nThreads - 1) / nThreads;

        //std::cout << "Platform Normal: " << platform.normal[0] << ", " << platform.normal[1] << "," << platform.normal[2] << "\n";

        cudaFunc << < nBlocks, nThreads >> > (minX, deltaX, minZ, deltaZ, nX, nZ, platform.normal[0], platform.normal[1], platform.normal[2], maxFrames);

        cudaMemcpyFromSymbol(&nPlatSolutionsCPU, nPlatSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);

        if (nPlatSolutionsCPU > 0) {
            printf("---------------------------------------\nTesting Normal: %g, %g, %g\n        Index: %d, %d, %d, %d\n", normX, normY, normZ, g, h, i, j);
            printf("        # Platform Solutions: %d\n", nPlatSolutionsCPU);

            if (nPlatSolutionsCPU > MAX_PLAT_SOLUTIONS) {
                fprintf(stderr, "Warning: Number of platform solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                nPlatSolutionsCPU = MAX_PLAT_SOLUTIONS;
            }

            nBlocks = (nPlatSolutionsCPU + nThreads - 1) / nThreads;

            //check_platform_solutions_for_the_right_one << < nBlocks, nThreads >> > ();

            cudaMemcpyToSymbol(nUpwarpSolutions, &nUpwarpSolutionsCPU, sizeof(int), 0, cudaMemcpyHostToDevice);

            find_upwarp_solutions << < nBlocks, nThreads >> > (1000000000.0f);

            cudaDeviceSynchronize();

            cudaMemcpyFromSymbol(&nUpwarpSolutionsCPU, nUpwarpSolutions, sizeof(int), 0, cudaMemcpyDeviceToHost);
        }
        else
        {
            printf("No platform solutions found for normal: %f, %f, %f.\n", normX, normY, normZ);
        }

        if (nUpwarpSolutionsCPU > 0) {
            if (nUpwarpSolutionsCPU > MAX_UPWARP_SOLUTIONS) {
                fprintf(stderr, "Warning: Number of upwarp solutions for this normal has been exceeded. No more solutions for this normal will be recorded. Increase the internal maximum to prevent this from happening.\n");
                nUpwarpSolutionsCPU = MAX_UPWARP_SOLUTIONS;
            }

            printf("        # Upwarp Solutions: %d\n", nUpwarpSolutionsCPU);

            nBlocks = (nUpwarpSolutionsCPU + nThreads - 1) / nThreads;

            struct PlatformSolution* platSolutionsCPU = (struct PlatformSolution*)std::malloc(nPlatSolutionsCPU * sizeof(struct PlatformSolution));

            cudaMemcpyFromSymbol(platSolutionsCPU, platSolutions, nPlatSolutionsCPU * sizeof(struct PlatformSolution), 0, cudaMemcpyDeviceToHost);

            struct UpwarpSolution* upwarpSolutionsCPU = (struct UpwarpSolution*)std::malloc(nUpwarpSolutionsCPU * sizeof(struct UpwarpSolution));

            cudaMemcpyFromSymbol(upwarpSolutionsCPU, upwarpSolutions, nUpwarpSolutionsCPU * sizeof(struct UpwarpSolution), 0, cudaMemcpyDeviceToHost);

            cudaDeviceSynchronize();

            float minSpeed = std::numeric_limits<float>::max();

            for (int i = 0; i < nUpwarpSolutionsCPU; ++i)
            {
                float puDist = sqrt((&upwarpSolutionsCPU[i])->pux* (&upwarpSolutionsCPU[i])->pux + (&upwarpSolutionsCPU[i])->puz * (&upwarpSolutionsCPU[i])->puz);
                float speed = 65536 * puDist / platSolutionsCPU[(&upwarpSolutionsCPU[i])->platformSolutionIdx].endNormal[1];

                if (speed < minSpeed)
                {
                    minSpeed = speed;
                }
            }

            minUpwarpSpeeds[((g * nSamplesNY + h) * nSamplesNX + i) * nSamplesNZ + j] = minSpeed;
            std::cout << "MinSpeed: " << minSpeed << "\n\n";

            free(platSolutionsCPU);
            free(upwarpSolutionsCPU);
        }
        else
        {
            printf("No upwarp solutions found.\n");
        }
    }
}





