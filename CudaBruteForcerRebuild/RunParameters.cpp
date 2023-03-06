#include "RunParameters.hpp"

int nThreads = 256;
size_t memorySize = 10000000;

bool computeMaxElevation = false;
bool computeMinUpwarp = false;

bool runHAUSolver = true;

bool useZXSum = false;
bool usePositiveZ = true;

bool stopAtUpwarp = false;
bool printOneOffSolutions = false;

int minQ1 = 1;
int maxQ1 = 1; // 2 for non-HAU, 4 for HAU, now set to 1
int minQ2 = 1;
int maxQ2 = 4;
int minQ3 = 1;
int maxQ3 = 1; // Default 2 for non-HAU, 4 for HAU, now set to 1

int nPUFrames = 3;
int maxFrames = 60; // 100 for non-HAU, 200 for HAU

float minNX = -0.2f;
float maxNX = -0.2f;
float minNZ = 0.4f;
float maxNZ = 0.4f;
float minNY = 0.8f;
float maxNY = 0.8f;
float minNZXSum = 0.0f;
float maxNZXSum = 0.0f;

int nSamplesNX = 1;
int nSamplesNZ = 1;
int nSamplesNY = 1;

float deltaX = 2.0f;
float deltaZ = 2.0f;

std::string normalsInput = "";

Vec3f platformPos = { -1945.0f, -3225.0f, -715.0f };

bool verbose = false;

