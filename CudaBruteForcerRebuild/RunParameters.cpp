#include "RunParameters.hpp"

int nThreads = 256;
size_t memorySize = 10000000;

bool computeMaxElevation = false;
bool computeMinUpwarp = false;

bool runHAUSolver = true;

bool useZXSum = true;
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

float minNX = -0.23570f;
float maxNX = -0.23506f;
float minNZ = 0.0f;
float maxNZ = 0.0f;
float minNY = 0.8330f;
float maxNY = 0.8335f;
float minNZXSum = 0.63690f;
float maxNZXSum = 0.63754f;

int nSamplesNX = 9;
int nSamplesNZ = 9;
int nSamplesNY = 6;

float deltaX = 0.5f;
float deltaZ = 0.5f;

std::string normalsInput = "";

Vec3f platformPos = { -1945.0f, -3225.0f, -715.0f };

bool verbose = false;

