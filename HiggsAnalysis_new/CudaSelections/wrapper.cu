//
//  wrapper.cpp
//  HiggsAnalysis_new
//
//  Created by Joona Havukainen on 5/28/19.
//  Copyright © 2019 Joona Havukainen. All rights reserved.
//

#include "wrapper.hpp"

void wrapper(float *array, int entries, int nVariables)
{
    float *d_array;
    cudaMalloc(&d_array, nVariables*entries*sizeof(float));
    cudaMemcpy(d_array, array, nVariables*entries*sizeof(float), cudaMemcpyHostToDevice);
    
    int blocks = (100000+1024)/1024;
    tauSelection<<blocks, 1024>>(d_array, nVariables);

    std::cout<<"Selection done"<<std::endl;
    cudaFree(d_array);
}
