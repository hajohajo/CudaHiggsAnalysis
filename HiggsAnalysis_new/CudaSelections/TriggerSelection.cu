//
//  TriggerSelection.cpp
//  HiggsAnalysis_new
//
//  Created by Joona Havukainen on 5/31/19.
//  Copyright © 2019 Joona Havukainen. All rights reserved.
//

__device__
bool L1METTrigger(float L1MET_x, float L1MET_y, float L1MET_cut)
{
    float L1MET = sqrtf(powf(L1MET_x, 2.f)+powf(L1MET_y, 2.f));
    return L1MET>L1MET_cut;
}

__global__
void triggerSelection(float *inputArray, bool *passedArray, float L1MetCut, int variablesPerEvent, int nEvents, int triggerIndex)
{
    int processIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int localIndex = processIndex * variablesPerEvent;
    
    if(processIndex<nEvents){
        passedArray[processIndex]=((bool)inputArray[localIndex+triggerIndex+2] && L1METTrigger(inputArray[localIndex+triggerIndex+0], inputArray[localIndex+triggerIndex+1], L1MetCut));
    }
}
