#include "RunParameters.hpp"

int nThreads = 256;
size_t memorySize = 10000000;

bool computeMaxElevation = false;
bool computeMinUpwarp = false;

bool runHAUSolver = true;

bool useZXSum = true;
bool usePositiveZ = true;

bool stopAtUpwarp = false;

int minQ1 = 1;
int maxQ1 = 1; // 2 for non-HAU, 4 for HAU, now set to 1
int minQ2 = 1;
int maxQ2 = 4;
int minQ3 = 1;
int maxQ3 = 1; // Default 2 for non-HAU, 4 for HAU, now set to 1

int nPUFrames = 3;
int maxFrames = 60; // 100 for non-HAU, 200 for HAU

float minNX = 0.0f;
float maxNX = 0.0f;
float minNZ = 0.0f;
float maxNZ = 0.0f;
float minNY = 0.0f;
float maxNY = 0.0f;
float minNZXSum = 0.0f;
float maxNZXSum = 0.0f;

int nSamplesNX = 1;
int nSamplesNZ = 1;
int nSamplesNY = 1;

float deltaX = 0.5f;
float deltaZ = 0.5f;

std::string normalsInput = "plotting/skyBlueNormalsQ2.txt";

Vec3f platformPos = { -1945.0f, -3225.0f, -715.0f };

bool verbose = true;

// Flag for whether to print sub-solutions (0 for none, 1 for second-to-last sub-solutions only, 2 for all sub-solutions).
int subSolutionPrintingMode = 1;
