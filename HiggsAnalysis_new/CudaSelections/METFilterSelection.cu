//
//  METFilterSelection.cpp
//  HiggsAnalysis_new
//
//  Created by Joona Havukainen on 5/31/19.
//  Copyright © 2019 Joona Havukainen. All rights reserved.
//

__global__
void metFilterSelection(float *inputArray, bool *passedArray, int variablesPerEvent, int nEvents, int metFilterIndex)
{
    int processIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int localIndex = processIndex * variablesPerEvent;
    
    if(processIndex<nEvents)
    {
        passedArray[processIndex + 0] = inputArray[localIndex + metFilterIndex + 0];
        passedArray[processIndex + 1] = inputArray[localIndex + metFilterIndex + 1];
        passedArray[processIndex + 2] = inputArray[localIndex + metFilterIndex + 2];
        passedArray[processIndex + 3] = inputArray[localIndex + metFilterIndex + 3];
        passedArray[processIndex + 4] = inputArray[localIndex + metFilterIndex + 4];
        passedArray[processIndex + 5] = inputArray[localIndex + metFilterIndex + 5;]

    }
    
}
