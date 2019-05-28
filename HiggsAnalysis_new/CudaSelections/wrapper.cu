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

void wrapper(float *array, int entries, int nVariables)
{
    float *d_array;
    cudaMalloc(&d_array, nVariables*entries*sizeof(float));
    cudaMemcpy(d_array, array, nVariables*entries*sizeof(float), cudaMemcpyHostToDevice);
    
    int blocks = (100000+1024)/1024;
    tauSelection<<<blocks, 1024>>>(d_array, entries);

    std::cout<<"Selection done"<<std::endl;
    cudaFree(d_array);
}
