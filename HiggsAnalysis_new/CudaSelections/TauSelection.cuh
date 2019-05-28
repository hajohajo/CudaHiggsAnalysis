#pragma once

extern __global__ void tauSelection(float *array, bool *passedArray, float *numericalArray, int variablesPerEvent, int tauIndex, int hltIndex, int nTaus);
