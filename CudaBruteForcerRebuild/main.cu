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
#include "RunParameters.hpp"
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
    return (int)(seconds / 3600.0);
}

int get_minutes_from_seconds(double seconds)
{
    int hour_seconds = get_hours_from_seconds(seconds) * 3600;
    return (int)((seconds - hour_seconds) / 60.0);
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
    std::string outFilePlatformHWRs = "output/platformHWRs_" + timestamp + ".bin";
    std::string outFileMinUpwarpSpeeds = "output/minUpwarpSpeeds_" + timestamp + ".bin";
    std::string outFileRunParams = "output/runParameters_" + timestamp + ".txt";

    nSamplesNX = minNX == maxNX ? 1 : nSamplesNX;
    nSamplesNY = minNY == maxNY ? 1 : nSamplesNY;

    if (!useZXSum)
        nSamplesNZ = minNZ == maxNZ ? 1 : nSamplesNZ;
    else
        nSamplesNZ = minNZXSum == maxNZXSum ? 1 : nSamplesNZ;

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
            printf("                                   If a list of normals is provided, then these parameters will define displacements from each normal.\n");
            printf("                                   Default: %g %g %d\n", minNX, maxNX, nSamplesNX);
            printf("-nz <min_nz> <max_nz> <n_samples>: Inclusive range of z normals to be considered, and the number of normals to sample.\n");
            printf("                                   If min_nz==max_nz then n_samples will be set to 1.\n");
            printf("                                   If a list of normals is provided, then these parameters will define displacements from each normal.\n");
            printf("                                   Default: %g %g %d\n", minNZ, maxNZ, nSamplesNZ);
            printf("-ny <min_ny> <max_ny> <n_samples>: Inclusive range of y normals to be considered, and the number of normals to sample.\n");
            printf("                                   If min_ny==max_ny then n_samples will be set to 1.\n");
            printf("                                   If a list of normals is provided, then these parameters will define displacements from each normal.\n");
            printf("                                   Default: %g %g %d\n", minNY, maxNY, nSamplesNY);
            printf("-dx <delta_x>: x coordinate spacing of positions on the platform.\n");
            printf("               Default: %g\n", deltaX);
            printf("-dz <delta_z>: z coordinate spacing of positions on the platform.\n");
            printf("               Default: %g\n", deltaZ);
            printf("-p <platform_x> <platform_y> <platform_z>: Position of the pyramid platform.\n");
            printf("                                           Default: %g %g %g\n", platformPos[0], platformPos[1], platformPos[2]);
            printf("-ni: Optional path to a list of normals around which to sample. If left empty, no list of normals is used, and samples are displaced from (0,0,0).\n");
            printf("    Default: %s\n", normalsInput.c_str());
            printf("-o: Path to the output file.\n");
            printf("    Default: %s\n", outFileSolutionData.c_str());
            printf("-rp: Path to the run parameters file.\n");
            printf("    Default: %s\n", outFileRunParams.c_str());
            printf("-sum <0 or 1>: Flag for whether to parameterize by Z or by ZXSum (0 for Z, 1 for ZXSum).\n");
            printf("    Default: %i\n", useZXSum);
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
        else if (!strcmp(argv[i], "-ni")) {
            normalsInput = argv[i + 1];
            i += 1;
        }
        else if (!strcmp(argv[i], "-o")) {
            outFileSolutionData = argv[i + 1];
            i += 1;
        }
        else if (!strcmp(argv[i], "-rp")) {
            outFileRunParams = argv[i + 1];
            i += 1;
        }
        else if (!strcmp(argv[i], "-sum")) {
            useZXSum = std::stoi(argv[i + 1]);
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

    std::cout << "    Initializing Floors..........\n";
    initialise_floors << < 1, 1 >> > ();

    set_platform_pos << < 1, 1 >> > (platformPos[0], platformPos[1], platformPos[2]);

    short* dev_tris;
    float* dev_norms;
    short* host_tris = (short*)std::malloc(18 * sizeof(short));
    float* host_norms = (float*)std::malloc(6 * sizeof(float));

    cudaMalloc((void**)&dev_tris, 18 * sizeof(short));
    cudaMalloc((void**)&dev_norms, 6 * sizeof(float));

    const float deltaNX = (nSamplesNX > 1) ? (maxNX - minNX) / (nSamplesNX - 1) : 0;
    const float deltaNZ = (nSamplesNZ > 1) ? (maxNZ - minNZ) / (nSamplesNZ - 1) : 0;
    const float deltaNY = (nSamplesNY > 1) ? (maxNY - minNY) / (nSamplesNY - 1) : 0;
    const float deltaNZXSum = (nSamplesNZ > 1) ? (maxNZXSum - minNZXSum) / (nSamplesNZ - 1) : 0;

    // Solution CSV output filestream
    std::ofstream wf(outFileSolutionData);

    // Run Parameters TXT output filestream
    std::ofstream wfrp(outFileRunParams);

    // Normal Stages binary output filestream
    std::ofstream wfns;

    // Height Wiggle Room binary output filestream
    std::ofstream wfphwrs;

    if (!computeMaxElevation && !computeMinUpwarp)
    {
        wfns = std::ofstream(outFileNormalStages, std::ios::out | std::ios::binary);
    }
    else
    {
        wfphwrs = std::ofstream(computeMaxElevation ? outFilePlatformHWRs : outFileMinUpwarpSpeeds, std::ios::out | std::ios::binary);
    }

    std::cout << "  Creating Normal Stage Array...\n";

    char* normalStages;
    float* platformHWRs;

    if (!computeMaxElevation && !computeMinUpwarp)
    {
        normalStages = (char*)std::calloc(sizeof(char), normals.size() * nSamplesNY * nSamplesNX * nSamplesNZ);
    }
    else
    {
        platformHWRs = (float*)std::calloc(sizeof(float), normals.size() * nSamplesNY * nSamplesNX * nSamplesNZ);
    }

    std::cout << "  Writing Run Parameters...\n";

    // Writing run parameters to separate .txt file
    wfrp << std::fixed;

    // TODO - REMOVE
    wfrp << "Platform Normal Tilt Data Collection Run"; // TODO - REMOVE

    wfrp << "Run Timestamp: " << timestamp << "\n\n";

    wfrp << "nThreads: " << nThreads << '\n';
    wfrp << "memorySize: " << memorySize << "\n\n";

    if (computeMaxElevation)
        wfrp << "Computing Max Elevation!\n\n";

    wfrp << "Is HAU-Aligned: " << runHAUSolver << "\n\n";

    wfrp << "Is ZXSum: " << useZXSum << "\n\n";

    if (useZXSum)
        wfrp << "Use Positive Z: " << usePositiveZ << "\n\n";

    wfrp << "MinQ1: " << minQ1 << '\n';
    wfrp << "MaxQ1: " << maxQ1 << '\n';
    wfrp << "MinQ2: " << minQ2 << '\n';
    wfrp << "MaxQ2: " << maxQ2 << '\n';
    wfrp << "MinQ3: " << minQ3 << '\n';
    wfrp << "MaxQ3: " << maxQ3 << "\n\n";

    wfrp << "nPUFrames: " << nPUFrames << '\n';
    wfrp << "maxFrames: " << maxFrames << "\n\n";

    wfrp << "minNX: " << minNX << '\n';
    wfrp << "maxNX: " << maxNX << '\n';
    wfrp << "minNY: " << minNY << '\n';
    wfrp << "maxNY: " << maxNY << '\n';

    if (!useZXSum)
    {
        wfrp << "minNZ: " << minNZ << '\n';
        wfrp << "maxNZ: " << maxNZ << "\n\n";
    }
    else
    {
        wfrp << "minNZXSum: " << minNZXSum << '\n';
        wfrp << "maxNZXSum: " << maxNZXSum << "\n\n";
    }

    wfrp << "nSamplesNX: " << nSamplesNX << '\n';
    wfrp << "nSamplesNY: " << nSamplesNY << '\n';
    wfrp << "nSamplesNZ: " << nSamplesNZ << "\n\n";

    wfrp << "deltaX: " << deltaX << '\n';
    wfrp << "deltaZ: " << deltaZ << "\n\n";

    wfrp << "NormalListPath: " << normalsInput << "\n\n";

    // TODO - UNCOMMENT
    /*
    if (runHAUSolver)
        setup_output_hau(wf);
    else
        setup_output_non_hau(wf);
    */

    // TODO - REMOVE
    wf << std::fixed;
    wf << "Normal X, Normal Y, Normal Z, ";
    wf << "End Normal X, End Normal Y, End Normal Z" << std::endl;

    std::unordered_map<uint64_t, PUSolution> puSolutionLookup;

    std::vector<Vec3f>::iterator normalIter = normals.begin();

    int current_normal = 0;
    float currentZXSum;
    bool currentPositive;

    std::cout << "Min XZSum: " << minNZXSum << "\n";
    std::cout << "Max XZSum: " << maxNZXSum << "\n";
    std::cout << "Delta X: " << deltaNX << "\n";
    std::cout << "Delta Y: " << deltaNY << "\n";
    std::cout << "Samples Z: " << nSamplesNZ << "\n";
    std::cout << "Delta XZSum: " << (maxNZXSum - minNZXSum) / (nSamplesNZ - 1) << "\n";

    std::cout << "\n  Startup Complete!\n\nStarting Bruteforcer...\n\n";

    for (normalIter; normalIter < normals.end(); normalIter++) {
        if (useZXSum)
        {
            currentZXSum = abs((*normalIter)[2]);

            currentPositive = (*normalIter)[2] > 0 ? true : (*normalIter)[2] < 0 ? false : usePositiveZ;
        }

        for (int h = 0; h < nSamplesNY; h++) {
            for (int i = 0; i < nSamplesNX; i++) {
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
                    else
                        run_common_bruteforcer(current_normal, h, i, j, normX, normY, normZ, host_tris, host_norms, dev_tris, dev_norms, puSolutionLookup, wf, normalStages);
                }
            }
        }

        current_normal++;
    }

    auto endTime = std::chrono::system_clock::now();

    free(host_tris);
    free(host_norms);
    cudaFree(dev_tris);
    cudaFree(dev_norms);
    wf.close();

    std::cout << "Writing output to file!\n";

    if (!computeMaxElevation && !computeMinUpwarp)
    {
        wfns.write(normalStages, normals.size() * nSamplesNY * nSamplesNX * nSamplesNZ);
        free(normalStages);
        wfns.close();
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

    wfrp << "Total Running Time: ";

    if (hours > 0)
    {
        wfrp << hours << " hr(s) ";
    }
    if (minutes > 0)
    {
        wfrp << minutes << " minute(s) ";
    }

    wfrp << seconds << " second(s)" << '\n';

    wfrp.close();

    return 0;
}
