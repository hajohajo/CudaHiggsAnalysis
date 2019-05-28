//
//  wrapper.cpp
//  HiggsAnalysis_new
//
//  Created by Joona Havukainen on 5/28/19.
//  Copyright © 2019 Joona Havukainen. All rights reserved.
//

#include <iostream>
#include "wrapper.h"
#include "TauSelection.cuh"

void wrapper(float *array, int entries, int nVariables, int tauIndex, int hltIndex, int nTaus)
{
    float *d_array, *d_numericalResults;
    bool *d_passedResults, *h_passedResults;
    float nFloatResults = 3;
    int nSelections = 3;
    
    h_passedResults = (bool*)calloc(entries*nSelections,sizeof(bool));
    cudaMalloc(&d_array, nVariables*entries*sizeof(float));
    cudaMalloc(&d_passedResults, entries*nSelections*sizeof(bool));
    cudaMalloc(&d_numericalResults, entries*nFloatResults*sizeof(float));
    cudaMemcpy(d_array, array, nVariables*entries*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_passedResults, 0, entries*nSelections*sizeof(bool));
    cudaMemset(d_numericalResults, 0.0, entries*nSelections*sizeof(float));

    int blocks = (100000+1024)/1024;
    tauSelection<<<blocks, 1024>>>(d_array, d_passedResults, d_numericalResults, nVariables, tauIndex, hltIndex, nTaus);

    std::cout<<std::endl;
    std::cout<<"Selection done"<<std::endl;
    cudaMemcpy(h_passedResults, d_passedResults, entries*nSelections*sizeof(bool), cudaMemcpyDeviceToHost);

    for(int i=0; i<entries*nSelections;i++)
    {
        if(i%nSelections==0)
        {
            std::cout<<std::endl;
        }
        std::cout<<h_passedResults[i];

    }
    std::cout<<std::endl;
    cudaFree(d_array);
    cudaFree(d_passedResults);
    cudaFree(d_numericalResults);

}
