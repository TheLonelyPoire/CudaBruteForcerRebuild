#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string>
#include <unordered_map>
#include <ctime>
#include <chrono>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#include "BruteforceFunctions.cuh"
#include "BruteforceVariables.cuh"
#include "CommonFunctions.cuh"
#include "Floors.cuh"
#include "HAUFunctions.cuh"
#include "NonHAUFunctions.cuh"
#include "RunParameters.hpp"
#include "SlideKickFunctions.cuh";
#include "SolutionStructs.cuh"

std::ofstream out_stream;

// Returns the timestamp of the argument as a string with the format MONTH_DAY_HOUR_MINUTE.
std::string get_timestamp(std::chrono::system_clock::time_point tp)
{
    std::time_t time_t = std::chrono::system_clock::to_time_t(tp);
    std::tm* tm = std::localtime(&time_t);
    
    return std::to_string(tm->tm_mon + 1) + "_" + std::to_string(tm->tm_mday) + "_" + std::to_string(tm->tm_hour) + "_" + std::to_string(tm->tm_min);
}

int get_hours_from_seconds(double seconds)
{
    return (int) (seconds / 3600.0);
}

int get_minutes_from_seconds(double seconds)
{
    int hour_seconds = get_hours_from_seconds(seconds) * 3600;
    return (int) ((seconds - hour_seconds) / 60.0);
}

/**
 * Reads a list of normals in from the file at the specified path.
 * 
 * The expected format for this file is a CSV. There should be one normal with three components per line. These components should be separated by commas.
 *
 * @param normals - A pointer to the list of normals. The function will clear the list and add all normals found in the file. 
 * @param filePath - The path of the file with the list of normals.
 * @return If the file is read without any problems, the function returns 0. If an error is encountered, the function will return 1.
 */
int read_normals_file(std::vector<Vec3f>& normals, std::string filePath)
{
    // This file-reading strategy is adapted from https://www.geeksforgeeks.org/csv-file-management-using-c/ 
    // with adjustments for the specifics

    normals = {};

    std::ifstream input_stream(filePath);

    if (input_stream.fail())
    {
        std::cout << "INVALID NORMAL LIST! Couldn't find the specified file!\n";
        return 1;
    }

    std::string word, temp;
    int line_counter = 0;

    while (input_stream >> temp)
    {
        normals.push_back({ 0,0,0 });

        std::stringstream words(temp);

        int word_counter = 0;

        while (std::getline(words, word, ','))
        {
            if (word_counter > 2)
            {
                std::cout << "INVALID NORMAL LIST! Found more than 3 elements in row " << line_counter << "!\n";
                return 1;
            }
            try {
                normals.back()[word_counter] = std::stof(word);
            }
            catch (const std::invalid_argument& ia)
            {
                std::cout << "INVALID NORMAL LIST! Could not convert word " << word_counter << " of row " << line_counter << " to a float!\n";
                return 1;
            }
            word_counter++;
        }

        if (word_counter < 3)
        {
            std::cout << "INVALID NORMAL LIST! Found less than 3 elements in row " << line_counter << "!\n";
            return 1;
        }

        line_counter++;
    }

    if (line_counter == 0)
    {
        std::cout << "INVALID NORMAL LIST! Didn't find any normals!\n";
        return 1;
    }

    return 0;
}

int main(int argc, char* argv[])
{   
    std::cout << "Bruteforcer Program Startup:\n";
    auto startTime = std::chrono::system_clock::now();
    std::string timestamp = get_timestamp(startTime);

    std::string outFileSolutionData = "output/solutionData_" + timestamp + ".csv";
    std::string outFileNormalStages = "output/normalStagesReached_" + timestamp + ".bin";
    std::string outFileFinalHeightDifferences = "output/finalHeightDifferences_" + timestamp + ".bin";
    std::string outFilePlatformHWRs = "output/platformHWRs_" + timestamp + ".bin";
    std::string outFileMinUpwarpSpeeds = "output/minUpwarpSpeeds_" + timestamp + ".bin";
    std::string outFileRunParams = "output/runParameters_" + timestamp + ".txt";

    nSamplesNX = minNX == maxNX ? 1 : nSamplesNX;
    nSamplesNY = minNY == maxNY ? 1 : nSamplesNY;

    if(!useZXSum)
        nSamplesNZ = minNZ == maxNZ ? 1 : nSamplesNZ;
    else
        nSamplesNZ = minNZXSum == maxNZXSum ? 1 : nSamplesNZ;


    // Process arguments passed to the bruteforcer (see CommonFunctions.cu for process_argument() definition)
    for (int i = 1; i < argc; i++) {
        process_argument(i, argv, outFileSolutionData, outFileRunParams);
    }

    if (nPUFrames != 3 && solverMode == 2) {
        fprintf(stderr, "Error: The slide kick brute forcer currently only supports 3 frame 10k routes. Value selected: %d.", nPUFrames);
        return 1;
    }

    const float deltaNX = (nSamplesNX > 1) ? (maxNX - minNX) / (nSamplesNX - 1) : 0;
    const float deltaNZ = (nSamplesNZ > 1) ? (maxNZ - minNZ) / (nSamplesNZ - 1) : 0;
    const float deltaNY = (nSamplesNY > 1) ? (maxNY - minNY) / (nSamplesNY - 1) : 0;
    const float deltaNZXSum = (nSamplesNZ > 1) ? (maxNZXSum - minNZXSum) / (nSamplesNZ - 1) : 0;

    // TODO - Adjust printing based on solver mode
    if (verbose) {
        printf("Max Frames: %d\n", maxFrames);
        printf("10K Frame 1 Q-Frames: (%d, %d)\n", minQ1, maxQ1);
        printf("10K Frame 2 Q-Frames: (%d, %d)\n", minQ2, maxQ2);
        printf("10K Frame 3 Q-Frames: (%d, %d)\n", minQ3, maxQ3);
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
        printf("Delta X: %f\n", deltaNX);
        if (useZXSum)
            printf("Delta ZXSum: %f\n", deltaNZXSum);
        else
            printf("Delta Z: %f\n", deltaNZ);
        printf("Delta Y: %f\n", deltaNY);
        if (normalsInput != "")
        {
            printf("Normals Input: ");
            printf(normalsInput.c_str());
            printf("\n");
        }
        printf("\n");
    }

    std::vector<Vec3f> normals = {};
    if (normalsInput == "")
    {
        normals.push_back({ 0,0,0 });
    }
    else
    {
        std::cout << "  Reading Normals from File...\n";

        // Returns 1 if an error is encountered, and this immediately returns 1 from main()
        if (read_normals_file(normals, normalsInput))
        {
            return 1;
        }
    }
    
    std::cout << "  Initializing Parameters........\n";

    std::cout << "    Initializing Reverse Atan....\n";
    init_reverse_atan << < 1, 1 >> > ();

    std::cout << "    Initializing Magnitude Set...\n";
    init_mag_set << < 1, 1 >> > ();

    std::cout << "    Initializing Camera Angles...\n";
    init_camera_angles << <1, 1 >> > ();

    std::cout << "    Initializing Floors..........\n";
    initialise_floors << < 1, 1 >> >();

    set_platform_pos << < 1, 1 >> >(platformPos[0], platformPos[1], platformPos[2]);

    short* dev_tris;
    float* dev_norms;
    short* host_tris = (short*)std::malloc(18 * sizeof(short));
    float* host_norms = (float*)std::malloc(6 * sizeof(float));

    cudaMalloc((void**)&dev_tris, 18 * sizeof(short));
    cudaMalloc((void**)&dev_norms, 6 * sizeof(float));

    // Solution CSV output filestream
    std::ofstream wf(outFileSolutionData);

    // Run Parameters TXT output filestream
    std::ofstream wfrp(outFileRunParams);

    // Normal Stages binary output filestream
    std::ofstream wfns;

    // Height Wiggle Room binary output filestream
    std::ofstream wfphwrs;

    // Final Height Difference binary output filestream
    std::ofstream wffhds;

    if (!computeMaxElevation && !computeMinUpwarp)
    {
        wfns = std::ofstream(outFileNormalStages, std::ios::out | std::ios::binary);
        if(solverMode == 1)
            wffhds = std::ofstream(outFileFinalHeightDifferences, std::ios::out | std::ios::binary);
    }
    else
    {
        wfphwrs = std::ofstream(computeMaxElevation ? outFilePlatformHWRs : outFileMinUpwarpSpeeds, std::ios::out | std::ios::binary);
    }

    std::cout << "  Creating Normal Stage Array...\n";

    char* normalStages;
    float* finalHeightDiffs;
    float* platformHWRs;

    if (!computeMaxElevation && !computeMinUpwarp)
    {
        normalStages = (char*)std::calloc(sizeof(char), normals.size() * nSamplesNY * nSamplesNX * nSamplesNZ);

        if (solverMode == 1)
        {
            finalHeightDiffs = (float*)std::malloc(sizeof(float) * normals.size() * nSamplesNY * nSamplesNX * nSamplesNZ);
            for (int i = 0; i < normals.size() * nSamplesNY * nSamplesNX * nSamplesNZ; ++i)
            {
                finalHeightDiffs[i] = MAX_HEIGHT_DIFFERENCE;
            }
        }
    }
    else
    {
        platformHWRs = (float*)std::calloc(sizeof(float), normals.size() * nSamplesNY * nSamplesNX * nSamplesNZ);
    }

    std::cout << "  Writing Run Parameters...\n";

    // Writing run parameters to separate .txt file
    write_run_parameters(wfrp, timestamp);

    wfrp.close();

    switch (solverMode)
    {
    case 0:
        setup_output_non_hau(wf);
        break;
    case 1:
        setup_output_hau(wf);
        break;
    case 2:
        setup_output_slide_kick(wf);
        break;
    }

    std::unordered_map<uint64_t, PUSolution> puSolutionLookup;

    std::vector<Vec3f>::iterator normalIter = normals.begin();

    int current_normal = 0;
    float currentZXSum;
    bool currentPositive;

    std::cout << "\n  Startup Complete!\n\nStarting Bruteforcer...\n\n";

    NonHAUSolStruct nonHAUSolutions;
    HAUSolStruct HAUSolutions;
    SKSolStruct slideKickSolutions;
    switch(solverMode)
    {
    case 0:
        init_solution_structs_non_hau(&nonHAUSolutions);
        break;
    case 1:
        init_solution_structs_hau(&HAUSolutions);
        break;
    case 2:
        init_solution_structs_sk(&slideKickSolutions);
        break;
    }


    short* floorPoints, *devFloorPoints;
    bool* squishEdges, *devSquishEdges;
    if (solverMode == 2)
    {
        floorPoints = (short*)std::malloc(4 * 3 * sizeof(short));
        cudaMalloc((void**)&devFloorPoints, 4 * 3 * sizeof(short));

        squishEdges = (bool*)std::malloc(4 * sizeof(bool));
        cudaMalloc((void**)&devSquishEdges, 4 * sizeof(bool));
    }

    float normal_offsets_cpu[4][3] = { {0.01f, -0.01f, 0.01f}, {-0.01f, -0.01f, 0.01f}, {-0.01f, -0.01f, -0.01f}, {0.01f, -0.01f, -0.01f} };

    for (normalIter; normalIter < normals.end(); normalIter++){
        if (useZXSum)
        {
            currentZXSum = abs((*normalIter)[2]);

            currentPositive = (*normalIter)[2] > 0 ? true : (*normalIter)[2] < 0 ? false : usePositiveZ;
        }

        for (int h = 0; h < nSamplesNY; h++) {
            for (int i = 0; i < nSamplesNX; i++) {

                if(subSolutionPrintingMode < 2)
                    std::cout << ((current_normal * nSamplesNY + h) * nSamplesNX + i) << " / " << normals.size() * nSamplesNY * nSamplesNX << "\r";

                for (int j = 0; j < nSamplesNZ; j++) {
                    float normX = (*normalIter)[0] + minNX + i * deltaNX;
                    float normY = (*normalIter)[1] + minNY + h * deltaNY;
                    float normZ;

                    if (useZXSum)
                    {
                        normZ = ((currentZXSum + minNZXSum + j * deltaNZXSum) - abs(normX)) * (currentPositive ? 1 : -1);
                    }
                    else
                    {
                        normZ = (*normalIter)[2] + minNZ + j * deltaNZ;
                    }

                    if (computeMaxElevation)
                        run_max_elevation_computations(current_normal, h, i, j, normX, normY, normZ, host_tris, host_norms, dev_tris, dev_norms, puSolutionLookup, wf, platformHWRs);
                    else if (computeMinUpwarp)
                        run_min_upwarp_speed_computations(current_normal, h, i, j, normX, normY, normZ, host_tris, host_norms, dev_tris, dev_norms, puSolutionLookup, wf, platformHWRs);
                    else if (solverMode == 0 || solverMode == 1)
                        run_common_bruteforcer(current_normal, h, i, j, normX, normY, normZ, host_tris, host_norms, dev_tris, dev_norms, puSolutionLookup, wf, normalStages, finalHeightDiffs);
                    else
                        run_slide_kick_bruteforcer(current_normal, h, i, j, normX, normY, normZ, host_tris, host_norms, dev_tris, dev_norms, slideKickSolutions, normal_offsets_cpu, floorPoints, devFloorPoints, squishEdges, devSquishEdges, wf, normalStages);
                }
            }
        }

        current_normal++;
    }

    auto endTime = std::chrono::system_clock::now();

    print_success << <1, 1 >> > ();

    free(host_tris);
    free(host_norms);
    cudaFree(dev_tris);
    cudaFree(dev_norms);

    switch(solverMode)
    {
    case 0:
        free_solution_pointers_non_hau(&nonHAUSolutions);
        break;
    case 1:
        free_solution_pointers_hau(&HAUSolutions);
        break;
    case 2:
        free(floorPoints);
        free(squishEdges);
        cudaFree(devFloorPoints);
        cudaFree(devSquishEdges);
        free_solution_pointers_sk(&slideKickSolutions);
    }

    wf.close();

    std::cout << "Writing output to file!\n";

    if (!computeMaxElevation && !computeMinUpwarp)
    {
        wfns.write(normalStages, normals.size() * nSamplesNY * nSamplesNX * nSamplesNZ);
        free(normalStages);
        wfns.close();

        if (solverMode == 1)
        {
            wffhds.write(reinterpret_cast<const char*>(finalHeightDiffs), normals.size() * nSamplesNY * nSamplesNX * nSamplesNZ * sizeof(float));
            free(finalHeightDiffs);
            wffhds.close();
        }
    }
    else
    {
        wfphwrs.write(reinterpret_cast<const char*>(platformHWRs), normals.size() * nSamplesNY * nSamplesNX * nSamplesNZ * sizeof(float));
        free(platformHWRs);
        wfphwrs.close();
    }

    std::chrono::duration<double> running_time = endTime - startTime;

    int hours = get_hours_from_seconds(running_time.count());
    int minutes = get_minutes_from_seconds(running_time.count());
    double seconds = running_time.count() - 3600 * hours - 60 * minutes;

    /*wfrp << "Total Running Time: ";
    
    if (hours > 0)
    {
        wfrp << hours << " hr(s) ";
    }
    if (minutes > 0)
    {
        wfrp << minutes << " minute(s) ";
    }
    
    wfrp << seconds << " second(s)" << '\n';*/

    return 0;
}
