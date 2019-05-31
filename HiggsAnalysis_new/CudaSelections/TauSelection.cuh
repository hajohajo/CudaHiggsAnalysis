#pragma once

extern __global__ void tauSelection(float *array, bool *passedArray, bool *passed, bool *selectedTaus, float *numericalArray, int variablesPerEvent, int tauIndex, int hltIndex, int nTaus, int nEvents);
