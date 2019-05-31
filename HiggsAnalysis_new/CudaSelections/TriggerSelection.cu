//
//  TriggerSelection.cpp
//  HiggsAnalysis_new
//
//  Created by Joona Havukainen on 5/31/19.
//  Copyright © 2019 Joona Havukainen. All rights reserved.
//

__global__
void triggerSelection(float *inputArray, bool *passedArray, int variablesPerEvent, int nEvents, int triggerIndex)
{
    int processIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int localIndex = processIndex * variablesPerEvent;
    
    if(processIndex<nEvents){
        passedArray[processIndex]=inputArray[localIndex+triggerIndex]
    }
}
