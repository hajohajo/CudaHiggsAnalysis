//
//  wrapper.cpp
//  HiggsAnalysis_new
//
//  Created by Joona Havukainen on 5/28/19.
//  Copyright © 2019 Joona Havukainen. All rights reserved.
//

#include <iostream>
#include "wrapper.h"
#include "TriggerSelection.cuh"
#include "TauSelection.cuh"

void wrapper(float *array, int entries, int nVariables, int tauIndex, int hltIndex, int nTaus)
{
    float *d_array, *d_numericalResults, *h_numericalResults;
    bool *d_passedResults, *h_passedResults, *d_selectedTaus, *h_selectedTaus, *d_passedTrigger, *h_passedTrigger;
    int nFloatResults = 3;
    int nSelections = 3;
    int nTrigger = 1;
    int triggerIndex = 3;
    float L1MetCut = 80.f;
    
    h_passedResults = (bool*)calloc(entries*nSelections,sizeof(bool));
    h_numericalResults = (float*)calloc(entries*nFloatResults,sizeof(float));
    h_selectedTaus = (bool*)calloc(entries*nTaus,sizeof(bool));
    h_passedTrigger = (bool*)calloc(entries*nTrigger, sizeof(bool));
    
    
    cudaMalloc(&d_array, nVariables*entries*sizeof(float));
    cudaMalloc(&d_passedResults, entries*nSelections*sizeof(bool));
    cudaMalloc(&d_numericalResults, entries*nFloatResults*sizeof(float));
    cudaMalloc(&d_selectedTaus, entries*nTaus*sizeof(bool));
    cudaMalloc(&d_passedTrigger, entries*nTrigger*sizeof(bool));
    cudaMemcpy(d_array, array, nVariables*entries*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_passedResults, 0, entries*nSelections*sizeof(bool));
    cudaMemset(d_numericalResults, 0.0, entries*nSelections*sizeof(float));
    cudaMemset(d_selectedTaus, 0, entries*nTaus*sizeof(bool));
    cudaMemset(d_passedTrigger, 0, entries*nTrigger*sizeof(bool));

//    int blocks = (100000+1024)/1024; //<<<blocks, 1024>>>
    triggerSelection<<<1, entries>>>(d_array, d_passedTrigger, L1MetCut, nVariables, entries, triggerIndex);
    tauSelection<<<1, entries>>>(d_array, d_passedResults, d_selectedTaus, d_numericalResults, nVariables, tauIndex, hltIndex, nTaus);
    std::cout<<std::endl;
    std::cout<<"Selection done"<<std::endl;
    cudaMemcpy(h_passedResults, d_passedResults, entries*nSelections*sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_numericalResults, d_numericalResults, entries*nFloatResults*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_selectedTaus, d_selectedTaus, entries*nTaus*sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_passedTrigger, d_passedTrigger, entries*nTrigger*sizeof(bool), cudaMemcpyDeviceToHost);

    for(int i=0; i<entries*nSelections;i++)
    {
        if(i%nSelections==0)
        {
            std::cout<<std::endl;
        }
        std::cout<<h_passedResults[i];

    }
    std::cout<<std::endl;
    
/*    for(int j=0; j<entries*nFloatResults; j++)
    {
        if(j%nFloatResults==0)
        {
            std::cout<<std::endl;
        }
        std::cout<<h_numericalResults[j]<<" ";
    }
    */
    
    for(int i = 0; i<entries*nTaus;i++)
    {
        if(i%nTaus==0)
        {
            std::cout<<std::endl;
        }
        std::cout<<h_selectedTaus[i]<<" ";
    }
    std::cout<<std::endl;

    for(int i = 0; i<entries*nTrigger;i++)
    {
        if(i%nTrigger==0)
        {
            std::cout<<std::endl;
        }
        std::cout<<h_passedTrigger[i]<<" ";
    }


    std::cout<<std::endl;
    cudaFree(d_array);
    cudaFree(d_passedResults);
    cudaFree(d_numericalResults);
    cudaFree(d_passedTrigger);
}
