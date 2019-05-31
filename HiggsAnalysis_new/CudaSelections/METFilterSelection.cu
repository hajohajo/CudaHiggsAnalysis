//
//  METFilterSelection.cpp
//  HiggsAnalysis_new
//
//  Created by Joona Havukainen on 5/31/19.
//  Copyright © 2019 Joona Havukainen. All rights reserved.
//

__global__
void metFilterSelection(float *inputArray, bool *passedArray, bool *passed, int variablesPerEvent, int nEvents, int metFilterIndex)
{
    int processIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int localIndex = processIndex * variablesPerEvent;
    int nMETFilters = 7;
    
    if(processIndex<nEvents)
    {
        bool pass = true;
        pass = pass && inputArray[localIndex + metFilterIndex + 0];
        passedArray[processIndex*nMETFilters + 0] = pass;
        
        pass = pass && inputArray[localIndex + metFilterIndex + 1];
        passedArray[processIndex*nMETFilters + 1] = pass;
        
        pass = pass && inputArray[localIndex + metFilterIndex + 2];
        passedArray[processIndex*nMETFilters + 2] = pass;
        
        pass = pass && inputArray[localIndex + metFilterIndex + 3];
        passedArray[processIndex*nMETFilters + 3] = pass;

        pass = pass && inputArray[localIndex + metFilterIndex + 4];
        passedArray[processIndex*nMETFilters + 4] = pass;

        pass = pass && inputArray[localIndex + metFilterIndex + 5];
        passedArray[processIndex*nMETFilters + 5] = pass;

        pass = pass && inputArray[localIndex + metFilterIndex + 6];
        passedArray[processIndex*nMETFilters + 6] = pass;

        passed[processIndex] = passed[processIndex] && pass;

    }
    
}
