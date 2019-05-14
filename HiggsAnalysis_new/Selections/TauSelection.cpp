//
//  TauSelection.cpp
//  HiggsAnalysis_new
//
//  Created by Joona Havukainen on 5/8/19.
//  Copyright © 2019 Joona Havukainen. All rights reserved.
//

#include "TauSelection.hpp"
#include <iostream>

bool TauSelection::makeSelection()
{
    return false;
};

TauSelection::TauSelection(uint currentIndex):Selection(currentIndex)
{
    int nTaus = 5;
    int nVariables = 3;
    int nEntries = 100;
    float *array = new float[nTaus*nVariables*nEntries];
    auto reader = [&](TTreeReader &reader)
    {
        TTreeReaderValue<std::vector<Double_t>> taus_pt(reader, "Taus_pt");
        TTreeReaderValue<std::vector<Double_t>> taus_eta(reader, "Taus_eta");
        TTreeReaderValue<std::vector<Double_t>> taus_phi(reader, "Taus_phi");
        while (reader.Next())
        {
            unsigned long i = reader.GetCurrentEntry();
            if(i>nEntries)
            {
                break;
            }
            for(int j = 0; j < nTaus; j++)
            {
                if(j<taus_pt->size())
                {
                    array[i*nTaus*nVariables + j * nVariables + 0] = taus_pt->at(j);
                    array[i*nTaus*nVariables + j * nVariables + 1] = taus_eta->at(j);
                    array[i*nTaus*nVariables + j * nVariables + 2] = taus_phi->at(j);
                }
                else{
                    array[i*nTaus*nVariables + j * nVariables + 0] = -99;
                    array[i*nTaus*nVariables + j * nVariables + 1] = -99;
                    array[i*nTaus*nVariables + j * nVariables + 2] = -99;
                }
            }
            std::cout<<"Entry: "<<i<<std::endl;
        }
        
    };
    
}

TauSelection::~TauSelection(){}
