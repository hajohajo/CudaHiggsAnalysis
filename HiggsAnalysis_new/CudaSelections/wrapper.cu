//
//  wrapper.cpp
//  HiggsAnalysis_new
//
//  Created by Joona Havukainen on 5/28/19.
//  Copyright © 2019 Joona Havukainen. All rights reserved.
//

#include <iostream>
#include <numeric>
#include <vector>
#include "wrapper.h"
#include "TriggerSelection.cuh"
#include "METFilterSelection.cuh"
#include "TauSelection.cuh"

int wrapper(float *array, int entries, int nVariables, int tauIndex, int hltIndex, int nTaus)
{
    float *d_array, *d_numericalResults, *h_numericalResults;
    bool *d_passedResults, *h_passedResults, *d_selectedTaus, *h_selectedTaus, *d_passedTrigger, *h_passedTrigger, *d_passedMETFilter, *h_passedMETFilter;
    bool *d_passed, *h_passed, *d_passedTaus, *h_passedTaus;
    int nFloatResults = 3;
    int nSelections = 3;
    int nTrigger = 1;
    int triggerIndex = 3;
    int metFilterIndex= 6;
    float L1MetCut = 80.f;
    int nMETFilter = 7;
    
    h_numericalResults = (float*)calloc(entries*nFloatResults,sizeof(float));
    h_selectedTaus = (bool*)calloc(entries*nTaus,sizeof(bool));
    h_passed = (bool*)calloc(entries*1,sizeof(bool));
    h_passedTrigger = (bool*)calloc(entries*nTrigger, sizeof(bool));
    h_passedMETFilter = (bool*)calloc(entries*nMETFilter, sizeof(bool));
    h_passedTaus = (bool*)calloc(entries*nTaus, sizeof(bool));
    
    
    cudaMalloc(&d_array, nVariables*entries*sizeof(float));
    cudaMalloc(&d_passedResults, entries*nSelections*sizeof(bool));
    cudaMalloc(&d_numericalResults, entries*nFloatResults*sizeof(float));
    cudaMalloc(&d_selectedTaus, entries*nTaus*sizeof(bool));
    cudaMalloc(&d_passedTrigger, entries*nTrigger*sizeof(bool));
    cudaMalloc(&d_passedMETFilter, entries*nMETFilter*sizeof(bool));
    cudaMalloc(&d_passed, entries*1*sizeof(bool));
    cudaMemcpy(d_array, array, nVariables*entries*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_passedResults, 0, entries*nSelections*sizeof(bool));
    cudaMemset(d_numericalResults, 0.0, entries*nSelections*sizeof(float));
    cudaMemset(d_selectedTaus, 0, entries*nTaus*sizeof(bool));
    cudaMemset(d_passedTrigger, 0, entries*nTrigger*sizeof(bool));
    cudaMemset(d_passedMETFilter, 0, entries*nMETFilter*sizeof(bool));
    cudaMemset(d_passed, 1, entries*1*sizeof(bool));

//    int blocks = (100000+1024)/1024; //<<<blocks, 1024>>>
    triggerSelection<<<4, 1024>>>(d_array, d_passedTrigger, d_passed, L1MetCut, nVariables, entries, triggerIndex);
    metFilterSelection<<<4, 1024>>>(d_array, d_passedMETFilter, d_passed, nVariables, entries, metFilterIndex);
    tauSelection<<<4, 1024>>>(d_array, d_passedResults, d_passed, d_selectedTaus, d_numericalResults, nVariables, tauIndex, hltIndex, nTaus, entries);

    cudaDeviceSynchronize();

//    std::cout<<std::endl;
//    std::cout<<"Selection done"<<std::endl;
//    cudaMemcpy(h_passedResults, d_passedResults, entries*nSelections*sizeof(bool), cudaMemcpyDeviceToHost);
//    cudaMemcpy(h_numericalResults, d_numericalResults, entries*nFloatResults*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_selectedTaus, d_selectedTaus, entries*nTaus*sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_passedTrigger, d_passedTrigger, entries*nTrigger*sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_passedMETFilter, d_passedMETFilter, entries*nMETFilter*sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_passed, d_passed, entries*1*sizeof(bool), cudaMemcpyDeviceToHost);

/*
    for(int i=0; i<entries*nSelections;i++)
    {
        if(i%nSelections==0)
        {
            std::cout<<std::endl;
        }
        std::cout<<h_passedResults[i];

    }
    std::cout<<std::endl;
*/
/*    for(int j=0; j<entries*nFloatResults; j++)
    {
        if(j%nFloatResults==0)
        {
            std::cout<<std::endl;
        }
        std::cout<<h_numericalResults[j]<<" ";
    }
    */
/*
    for(int i = 0; i<entries*nTaus;i++)
    {
        if(i%nTaus==0)
        {
            std::cout<<std::endl;
        }
        std::cout<<h_selectedTaus[i]<<" ";
    }
    std::cout<<std::endl;
*/
/*
    for(int i = 0; i<entries*nTrigger;i++)
    {
        if(i%nTrigger==0)
        {
            std::cout<<std::endl;
        }
        std::cout<<h_passedTrigger[i]<<" ";
    }
*/
/*
    for(int i = 0; i<entries*nMETFilter;i++)
    {
        if(i%nMETFilter==0)
        {
            std::cout<<std::endl;
        }
        std::cout<<h_passedMETFilter[i]<<" ";
    }
    
*/
/*    for(int i = 0; i<entries;i++)
    {
        std::cout<<std::endl;
        std::cout<<h_passed[i];
    }
    std::cout<<std::endl;
*/
    int sum = 0;
    for(int i = 0; i<entries;i++)
    {
        sum += h_passed[i];
    }

    std::cout<<sum<<" out of "<<entries<<std::endl;

    std::cout<<std::endl;
    cudaFree(d_array);
    cudaFree(d_passedResults);
    cudaFree(d_numericalResults);
    cudaFree(d_passedTrigger);
    cudaFree(d_passedMETFilter);
    cudaFree(d_passed);
    
    return sum;
}
