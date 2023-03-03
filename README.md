# CudaBruteForcerRebuild
A modified version of the bruteforcer developed by SpudY2K, which can be found here: https://github.com/SpudY2K/sm64-platform-max-tilt-brute-forcer/tree/master/platform_tilt_brute_forcer/platform_tilt_brute_forcer

The main modifications made were to allow for various types of data output (Run Parameters, Normal Stages, Height Wiggle Room, etc.) and some restructuring to avoid super long files.

I've also included the original bruteforcer's README below for reference.

## sm64-platform-max-tilt-brute-forcer
A tool for identifying valid position ranges for Mario to tilt the inverted pyramid platform to an upwarp-compatible normal.

A lot of this code came from previous brute forcers from folks like SpudY2K, AnthonyC4 and myself, as well as SM64 decomp.
I made changes to some files I copied over for this specific project.
Thanks to Modiseus and SpudY2K for the boundary distance logic. I made some edits there as well.

This tool is simple enough: given a starting platform normal, iterate over a range of starting Mario poaitions for 100 frames, and keep track of how close to the upwarp boundary Mario is able to tilt the normal to.
The assumption here is that Mario already has PU speed and so is not able to snap to the platform's floor each frame, and is completely at the mercy of the platform displacement error to continue tilting the platform. Eventually the accumulated displacement error will prevent Mario from tilting the platform further, usually sooner rather than later.
For each starting position, Mario starts at the height of the platform floor (which he would snap up to if he approaches from below it)
The best boundary distance (usually negative) is recorded for each starting position. Filtering only positive values for these will return the upwarp-compatible starting positions.
Values are returned in a csv for convenience.

Developed in VS 2019
