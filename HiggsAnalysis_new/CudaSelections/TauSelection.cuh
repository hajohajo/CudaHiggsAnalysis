#pragma once

extern __global__ void tauSelection(float *array, bool *passedArray, bool *selectedTaus, float *numericalArray, int variablesPerEvent, int tauIndex, int hltIndex, int nTaus);
