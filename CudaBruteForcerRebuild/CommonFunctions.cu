#include "CommonFunctions.cuh"

#include "math.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"
//#include "device_types.h"

#include "Platform.cuh"
#include "vmath.hpp"

#include "BruteforceVariables.cuh"
#include "Floors.cuh"
#include "RunParameters.hpp"


void write_run_parameters(std::ofstream& wfrp, std::string timestamp)
{
    wfrp << std::fixed;

    wfrp << "Run Timestamp: " << timestamp << "\n\n";

    wfrp << "nThreads: " << nThreads << '\n';
    wfrp << "memorySize: " << memorySize << "\n\n";

    if (computeMaxElevation)
        wfrp << "Computing Max Elevation!\n\n";

    wfrp << "Solver Mode: " << solverMode << "\n\n";

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
    wfrp << "nSamplesNZ: " << nSamplesNZ << '\n';
    wfrp << "nSamplesNY: " << nSamplesNY << "\n\n";

    wfrp << "deltaX: " << deltaX << '\n';
    wfrp << "deltaZ: " << deltaZ << "\n\n";

    wfrp << "NormalListPath: " << normalsInput << "\n\n";

    wfrp << "Platform Position: " << platformPos[0] << ", " << platformPos[1] << ", " << platformPos[2] << "\n";
}

void print_help(std::string &default_output_path, std::string &default_run_params_path)
{
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
    printf("                                   ONLY USED IF -sum IS SET TO 0.\n");
    printf("                                   If min_nz==max_nz then n_samples will be set to 1.\n");
    printf("                                   If a list of normals is provided, then these parameters will define displacements from each normal.\n");
    printf("                                   Default: %g %g %d\n", minNZ, maxNZ, nSamplesNZ);
    printf("-nzxsum <min_nzxsum> <max_nzxsum> <n_samples>: Inclusive range of zxsum normals to be considered, and the number of normals to sample.\n");
    printf("                                               ONLY USED IF -sum IS SET TO 1.\n");
    printf("                                               If min_nz==max_nz then n_samples will be set to 1.\n");
    printf("                                               If a list of normals is provided, then these parameters will define displacements from each normal.\n");
    printf("                                               Default: %g %g %d\n", minNZ, maxNZ, nSamplesNZ);
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
    printf("-solver <0, 1, or 2>: Flag for which solver to use (0 for non-HAU-Aligned, 1 for HAU-Aligned, 2 for Slide Kick).\n");
    printf("                      Default: %i\n", solverMode);
    printf("-ni: Optional path to a list of normals around which to sample. If left empty, no list of normals is used, and samples are displaced from (0,0,0).\n");
    printf("    Default: %s\n", normalsInput.c_str());
    printf("-o: Path to the output file.\n");
    printf("    Default: %s\n", default_output_path.c_str());
    printf("-rp: Path to the run parameters file.\n");
    printf("     Default: %s\n", default_run_params_path.c_str());
    printf("-sum <0 or 1>: Flag for whether to parameterize by Z or by ZXSum (0 for Z, 1 for ZXSum).\n");
    printf("               Default: %i\n", useZXSum);
    printf("-posZ <0 or 1>: Flag for whether to use postive Z or negative Z (0 for -Z, 1 for +Z).\n");
    printf("                Only used when parameterizing by ZXSum instead of Z.\n");
    printf("                Default: %i\n", usePositiveZ);
    printf("-ssp <0, 1, or 2>: Printing mode for subsolutions (0 for no subsolution printing, 1 for minimal printing, 2 for full printing).\n");
    printf("                   Default: %i\n", subSolutionPrintingMode);
    printf("-t <threads>: Number of CUDA threads to assign to the program.\n");
    printf("              Default: %d\n", nThreads);
    printf("-m <memory>: Amount of GPU memory to assign to the program.\n");
    printf("             Default: %d\n", memorySize);
    printf("-v: Verbose mode. Prints all parameters used in brute force.\n");
    printf("    Default: off\n");
    printf("-h --help: Prints this text.\n");
    exit(0);
}

void process_argument(int& i, char* argv[], std::string& outFileSolutionData, std::string& outFileRunParams)
{
    if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
        print_help(outFileSolutionData, outFileRunParams);
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
    else if (!strcmp(argv[i], "-nzxsum")) {
        minNZXSum = std::stof(argv[i + 1]);
        maxNZXSum = std::stof(argv[i + 2]);

        if (minNZXSum == maxNZXSum) {
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
    else if (!strcmp(argv[i], "-solver")) {
        solverMode = std::stoi(argv[i + 1]);
        i += 1;
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
    else if (!strcmp(argv[i], "-posZ")) {
        usePositiveZ = std::stoi(argv[i + 1]);
        i += 1;
    }
    else if (!strcmp(argv[i], "-ssp")) {
        subSolutionPrintingMode = std::stoi(argv[i + 1]);
        i += 1;
    }
    else if (!strcmp(argv[i], "-v")) {
        verbose = true;
    }
}

__global__ void print_success() {
    printf("CUDA code completed successfully.\n");
}

// This was taken from the following StackOverflow post: https://stackoverflow.com/a/51549250
__device__ float atomicMinFloat(float* addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

__device__ bool check_inbounds(const float* mario_pos) {
    short x_mod = (short)(int)mario_pos[0];
    short y_mod = (short)(int)mario_pos[1];
    short z_mod = (short)(int)mario_pos[2];

    return (abs(x_mod) < 8192 & abs(y_mod) < 8192 & abs(z_mod) < 8192);
}

__global__ void set_squish_ceilings(float n0, float n1, float n2, float n3) {
    squishCeilings[0] = n0 > -0.5;
    squishCeilings[1] = n1 > -0.5;
    squishCeilings[2] = n2 > -0.5;
    squishCeilings[3] = n3 > -0.5;
}

__global__ void set_platform_pos(float x, float y, float z) {
    platform_pos[0] = x;
    platform_pos[1] = y;
    platform_pos[2] = z;
}

__global__ void set_platform_normal(float nx, float ny, float nz) {
    platformNormal[0] = nx;
    platformNormal[1] = ny;
    platformNormal[2] = nz;
}

__global__ void calculate_10k_multipliers(int minQ1Q2, int maxQ1Q2, int minQ3, int maxQ3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < 112) {
        int temp = idx;
        int q1q2 = (temp % 7) + 2;
        temp = temp / 7;
        int q3 = (temp % 4) + 1;
        temp = temp / 4;
        int f = temp % 2;
        temp = temp / 2;
        int o = temp;

        if (q1q2 >= minQ1Q2 && q1q2 <= maxQ1Q2 && q3 >= minQ3 && q3 <= maxQ3) {
            tenKMultipliers[idx] = -(startNormals[f][1] + (double)q1q2 - 1.0) / ((o == 0 ? oneUpPlatformNormalYRight : oneUpPlatformNormalYLeft) + (double)q3 - 1.0);
        }
        else {
            tenKMultipliers[idx] = NAN;
        }
    }
}

__global__ void init_reverse_atan() {

    for (int i = 0; i < 8192; i++) {
        int angle = (65536 + gArctanTableG[i]) % 65536;
        gReverseArctanTable[angle] = i;
    }
}

__global__ void set_start_triangle(short* tris, float* norms) {
    for (int x = 0; x < 2; x++) {
        for (int y = 0; y < 3; y++) {
            startTriangles[x][y][0] = tris[9 * x + 3 * y];
            startTriangles[x][y][1] = tris[9 * x + 3 * y + 1];
            startTriangles[x][y][2] = tris[9 * x + 3 * y + 2];
            startNormals[x][y] = norms[3 * x + y];
        }
    }
}

__device__ int16_t atan2_lookupG(float z, float x) {
    int16_t angle = 0;

    if (x == 0) {
        angle = gArctanTableG[0];
    }
    else {
        angle = gArctanTableG[uint16_t(float(float(z / x) * 1024.0 + 0.5))];
    }

    return angle;
}

__device__ int16_t atan2sG(float z, float x) {
    int16_t angle = 0;

    if (x >= 0) {
        if (z >= 0) {
            if (z >= x) {
                angle = atan2_lookupG(x, z);
            }
            else {
                angle = 0x4000 - atan2_lookupG(z, x);
            }
        }
        else {
            z = -z;

            if (z < x) {
                angle = 0x4000 + atan2_lookupG(z, x);
            }
            else {
                angle = 0x8000 - atan2_lookupG(x, z);
            }
        }
    }
    else {
        x = -x;

        if (z < 0) {
            z = -z;

            if (z >= x) {
                angle = 0x8000 + atan2_lookupG(x, z);
            }
            else {
                angle = 0xC000 - atan2_lookupG(z, x);
            }
        }
        else {
            if (z < x) {
                angle = 0xC000 + atan2_lookupG(z, x);
            }
            else {
                angle = -atan2_lookupG(x, z);
            }
        }
    }

    return ((angle + 32768) % 65536) - 32768;
}

__device__ float find_closest_mag(float target) {
    int minIdx = -1;
    int maxIdx = magCount;

    while (maxIdx > minIdx + 1) {
        int midIdx = (maxIdx + minIdx) / 2;

        if (target < magSet[midIdx]) {
            maxIdx = midIdx;
        }
        else {
            minIdx = midIdx;
        }
    }

    if (minIdx == -1) {
        return magSet[maxIdx];
    }
    else if (maxIdx == magCount) {
        return magSet[minIdx];
    }
    else if (target - magSet[minIdx] < magSet[maxIdx] - target) {
        return magSet[minIdx];
    }
    else {
        return magSet[maxIdx];
    }
}

__global__ void init_mag_set() {
    bool magCheck[4097];

    for (int i = 0; i <= 4096; i++) {
        magCheck[i] = false;
    }

    for (int x = -128; x < 128; x++) {
        for (int y = -128; y < 128; y++) {
            int xS;
            if (x < 8) {
                if (x > -8) {
                    xS = 0;
                }
                else {
                    xS = x + 6;
                }
            }
            else {
                xS = x - 6;
            }
            int yS;
            if (y < 8) {
                if (y > -8) {
                    yS = 0;
                }
                else {
                    yS = y + 6;
                }
            }
            else {
                yS = y - 6;
            }

            int mag2 = xS * xS + yS * yS;
            mag2 = mag2 > 4096 ? 4096 : mag2;

            magCheck[mag2] = true;
        }
    }

    for (int i = 0; i <= 4096; i++) {
        if (magCheck[i]) {
            float mag = sqrtf((float)i);
            mag = (mag / 64.0f) * (mag / 64.0f) * 32.0f;
            magSet[magCount] = mag;
            magCount++;
        }
    }
}

__global__ void init_camera_angles() {
    for (int i = 0; i < 65536; i += 16) {
        int angle = atan2sG(gCosineTableG[i >> 4], gSineTableG[i >> 4]);
        angle = (65536 + angle) % 65536;

        validCameraAngle[angle] = true;
    }
}

__device__ int atan2b(double z, double x) {
    double A = 65536 * atan2(x, z) / (2 * M_PI);
    A = fmod(65536.0 + A, 65536.0);
    int lower = 0;
    int upper = 8192;

    while (upper - lower > 1) {
        int mid = (upper + lower) / 2;

        if (fmod(65536.0 + gArctanTableG[mid], 65536.0) > A) {
            upper = mid;
        }
        else {
            lower = mid;
        }
    }

    return lower;
}

__device__ int calculate_camera_yaw(float* currentPosition, float* lakituPosition) {
    short baseCameraYaw = -16384;
    float baseCameraDist = 1400.0;
    short baseCameraPitch = 0x05B0;
    short baseCameraFaceAngle = 24576;

    SurfaceG* floor;
    float floorY;

    float xOff = currentPosition[0] + gSineTableG[((65536 + (int)baseCameraYaw) % 65536) >> 4] * 40.f;
    float zOff = currentPosition[2] + gCosineTableG[((65536 + (int)baseCameraYaw) % 65536) >> 4] * 40.f;
    float offPos[3] = { xOff, currentPosition[1], zOff };

    int floorIdx = find_floor(offPos, &floor, floorY, floorsG, total_floorsG);
    floorY = floorY - currentPosition[1];

    if (floorIdx != -1) {
        if (floorY > 0) {
            if (!(floor->normal[2] == 0.f && floorY < 100.f)) {
                baseCameraPitch += atan2sG(40.f, floorY);
            }
        }
    }

    baseCameraPitch = baseCameraPitch + 2304;

    float cameraPos[3] = { currentPosition[0] + baseCameraDist * gCosineTableG[((65536 + (int)baseCameraPitch) % 65536) >> 4] * gSineTableG[((65536 + (int)baseCameraYaw) % 65536) >> 4],
                       currentPosition[1] + 125.0f + baseCameraDist * gSineTableG[((65536 + (int)baseCameraPitch) % 65536) >> 4],
                       currentPosition[2] + baseCameraDist * gCosineTableG[((65536 + (int)baseCameraPitch) % 65536) >> 4] * gCosineTableG[((65536 + (int)baseCameraYaw) % 65536) >> 4]
    };

    float pan[3] = { 0, 0, 0 };
    float temp[3] = { 0, 0, 0 };

    // Get distance and angle from camera to Mario.
    float dx = currentPosition[0] - cameraPos[0];
    float dy = currentPosition[1] + 125.0f;
    float dz = currentPosition[2] - cameraPos[2];

    float cameraDist = sqrtf(dx * dx + dy * dy + dz * dz);
    float cameraPitch = atan2sG(sqrtf(dx * dx + dz * dz), dy);
    float cameraYaw = atan2sG(dz, dx);

    // The camera will pan ahead up to about 30% of the camera's distance to Mario.
    pan[2] = gSineTableG[0xC0] * cameraDist;

    temp[0] = pan[0];
    temp[1] = pan[1];
    temp[2] = pan[2];

    pan[0] = temp[2] * gSineTableG[((65536 + (int)baseCameraFaceAngle) % 65536) >> 4] + temp[0] * gCosineTableG[((65536 + (int)baseCameraFaceAngle) % 65536) >> 4];
    pan[2] = temp[2] * gCosineTableG[((65536 + (int)baseCameraFaceAngle) % 65536) >> 4] + temp[0] * gSineTableG[((65536 + (int)baseCameraFaceAngle) % 65536) >> 4];

    // rotate in the opposite direction
    cameraYaw = -cameraYaw;

    temp[0] = pan[0];
    temp[1] = pan[1];
    temp[2] = pan[2];

    pan[0] = temp[2] * gSineTableG[((65536 + (int)cameraYaw) % 65536) >> 4] + temp[0] * gCosineTableG[((65536 + (int)cameraYaw) % 65536) >> 4];
    pan[2] = temp[2] * gCosineTableG[((65536 + (int)cameraYaw) % 65536) >> 4] + temp[0] * gSineTableG[((65536 + (int)cameraYaw) % 65536) >> 4];

    // Only pan left or right
    pan[2] = 0.f;

    cameraYaw = -cameraYaw;

    temp[0] = pan[0];
    temp[1] = pan[1];
    temp[2] = pan[2];

    pan[0] = temp[2] * gSineTableG[((65536 + (int)cameraYaw) % 65536) >> 4] + temp[0] * gCosineTableG[((65536 + (int)cameraYaw) % 65536) >> 4];
    pan[2] = temp[2] * gCosineTableG[((65536 + (int)cameraYaw) % 65536) >> 4] + temp[0] * gSineTableG[((65536 + (int)cameraYaw) % 65536) >> 4];

    float cameraFocus[3] = { currentPosition[0] + pan[0], currentPosition[1] + 125.0f + pan[1], currentPosition[2] + pan[2] };

    dx = cameraFocus[0] - lakituPosition[0];
    dy = cameraFocus[1] - lakituPosition[1];
    dz = cameraFocus[2] - lakituPosition[2];

    cameraDist = sqrtf(dx * dx + dy * dy + dz * dz);
    cameraPitch = atan2sG(sqrtf(dx * dx + dz * dz), dy);
    cameraYaw = atan2sG(dz, dx);

    if (cameraPitch > 15872) {
        cameraPitch = 15872;
    }
    if (cameraPitch < -15872) {
        cameraPitch = -15872;
    }

    cameraFocus[0] = lakituPosition[0] + cameraDist * gCosineTableG[((65536 + (int)cameraPitch) % 65536) >> 4] * gSineTableG[((65536 + (int)cameraYaw) % 65536) >> 4];
    cameraFocus[1] = lakituPosition[1] + cameraDist * gSineTableG[((65536 + (int)cameraPitch) % 65536) >> 4];
    cameraFocus[2] = lakituPosition[2] + cameraDist * gCosineTableG[((65536 + (int)cameraPitch) % 65536) >> 4] * gCosineTableG[((65536 + (int)cameraYaw) % 65536) >> 4];

    return atan2sG(lakituPosition[2] - cameraFocus[2], lakituPosition[0] - cameraFocus[0]);
}

__device__ void platform_logic_gpu(float* platform_normal, float* mario_pos, short(&triangles)[2][3][3], float(&normals)[2][3], float(&mat)[4][4]) {
    float dx;
    float dy;
    float dz;
    float d;

    float dist[3];
    float posBeforeRotation[3];
    float posAfterRotation[3];

    // Mario's position
    float mx = mario_pos[0];
    float my = mario_pos[1];
    float mz = mario_pos[2];

    dist[0] = mx - (float)platform_pos[0];
    dist[1] = my - (float)platform_pos[1];
    dist[2] = mz - (float)platform_pos[2];

    mat[1][0] = platform_normal[0];
    mat[1][1] = platform_normal[1];
    mat[1][2] = platform_normal[2];

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

    mat[3][0] = platform_pos[0];
    mat[3][1] = platform_pos[1];
    mat[3][2] = platform_pos[2];
    mat[0][3] = 0.0f;
    mat[1][3] = 0.0f;
    mat[2][3] = 0.0f;
    mat[3][3] = 1.0f;

    for (int i = 0; i < 3; i++) {
        posBeforeRotation[i] = mat[0][i] * dist[0] + mat[1][i] * dist[1] + mat[2][i] * dist[2];
    }

    dx = mx - (float)platform_pos[0];
    dy = 500.0f;
    dz = mz - (float)platform_pos[2];
    d = sqrtf(dx * dx + dy * dy + dz * dz);

    // Normalizing
    d = 1.0 / d;
    dx *= d;
    dy *= d;
    dz *= d;

    // Approach the normals by 0.01f towards the new goal, then create a transform matrix and orient the object. 
    // Outside of the other conditionals since it needs to tilt regardless of whether Mario is on.
    platform_normal[0] = (platform_normal[0] <= dx) ? ((dx - platform_normal[0] < 0.01f) ? dx : (platform_normal[0] + 0.01f)) : ((dx - platform_normal[0] > -0.01f) ? dx : (platform_normal[0] - 0.01f));
    platform_normal[1] = (platform_normal[1] <= dy) ? ((dy - platform_normal[1] < 0.01f) ? dy : (platform_normal[1] + 0.01f)) : ((dy - platform_normal[1] > -0.01f) ? dy : (platform_normal[1] - 0.01f));
    platform_normal[2] = (platform_normal[2] <= dz) ? ((dz - platform_normal[2] < 0.01f) ? dz : (platform_normal[2] + 0.01f)) : ((dz - platform_normal[2] > -0.01f) ? dz : (platform_normal[2] - 0.01f));

    mat[1][0] = platform_normal[0];
    mat[1][1] = platform_normal[1];
    mat[1][2] = platform_normal[2];

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

    mat[3][0] = platform_pos[0];
    mat[3][1] = platform_pos[1];
    mat[3][2] = platform_pos[2];
    mat[0][3] = 0.0f;
    mat[1][3] = 0.0f;
    mat[2][3] = 0.0f;
    mat[3][3] = 1.0f;

    for (int i = 0; i < 3; i++) {
        posAfterRotation[i] = mat[0][i] * dist[0] + mat[1][i] * dist[1] + mat[2][i] * dist[2];
    }

    mx += posAfterRotation[0] - posBeforeRotation[0];
    my += posAfterRotation[1] - posBeforeRotation[1];
    mz += posAfterRotation[2] - posBeforeRotation[2];
    mario_pos[0] = mx;
    mario_pos[1] = my;
    mario_pos[2] = mz;
}