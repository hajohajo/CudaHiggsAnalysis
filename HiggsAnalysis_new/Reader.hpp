//
//  Reader.hpp
//  HiggsAnalysis_new
//
//  Created by Joona Havukainen on 5/13/19.
//  Copyright © 2019 Joona Havukainen. All rights reserved.
//

//*******************************************************************
//This class is responsible for reading the needed variables from the
//ROOT trees and storing them to arrays that will be dispatched to the
//GPUs.
//
//Since the arrays sent must be fixed size, for variables that vary
//from event to event, some empty buffers are needed (i.e. always store
//five taus even though event rarely has that many)
//*******************************************************************

//The definite list of variables and their indices:
//nTaus = 10, each tau has 11 variables. So for indices 0-109, the pattern is:
// 0: tau pT
// 1: tau eta
// 2: tau phi
// 3: tau lChTrkPt
// 4: tau lChTrkEta
// 5: tau nProngs
// 6: tau decayMode
// 7: tau decayModeFinding
// 8: tau againstElectronTightMVA6
// 9: tau againstMuonLoose3
// 10: tau isolation

//MET
//nVariables = 2, indices 110-111
//110: MET_Type1_x
//111: MET_Type1_y

#ifndef Reader_hpp
#define Reader_hpp

#include <stdio.h>
#include <ROOT/TTreeProcessorMT.hxx>

class Reader
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
    Reader();
    ~Reader();
    void readToArray();
    void setFiles(std::vector<std::string> files);
    float* getArrayToGPU();
    int getNumberOfVariables();
    int getBatchSize();
    
    int getTauIndex();
    int getHltIndex();
    int getNTaus();
};

#endif /* Reader_hpp */
