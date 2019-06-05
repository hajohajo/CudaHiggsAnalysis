//
//  Reader_CPU.hpp
//  HiggsAnalysis_new
//
//  Created by Joona Havukainen on 6/5/19.
//  Copyright © 2019 Joona Havukainen. All rights reserved.
//

#ifndef Reader_CPU_hpp
#define Reader_CPU_hpp

#include <stdio.h>
#include <vector>
#include <string>
#include "CudaSelections/wrapper.h"
class Reader_CPU
{
private:
    int nCores;
    int numberOfVariables;
    int batchSize;
    int entryIndex;
    float* arrayToGPU;
    std::vector<std::string> inputFiles;
    
    //Global variables
    int globalIndex;
    int globalVariables;
    
    //Trigger
    int triggerIndex;
    int triggerVariables;
    
    //MET filters
    int metFilterIndex;
    int metFilterVariables;
    
    //Taus
    int tauIndex;
    int nTaus;
    int variablesPerTau;
    
    //HLT taus
    int hltIndex;
    int nHLTTaus;
    int variablesPerHLTTau;
    
    //Jets
    int jetIndex;
    int nJets;
    int variablesPerJet;
    
    //MET
    int metIndex;
    int metVariables;
    
    
public:
    Reader_CPU();
    ~Reader_CPU();
    void readToArray();
    void setFiles(std::vector<std::string> files);
    float* getArrayToGPU();
    int getNumberOfVariables();
    int getBatchSize();
    
    int getTauIndex();
    int getHltIndex();
    int getNTaus();
};
#endif /* Reader_CPU_hpp */
