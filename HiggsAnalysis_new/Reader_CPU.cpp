//
//  Reader_CPU.cpp
//  HiggsAnalysis_new
//
//  Created by Joona Havukainen on 6/5/19.
//  Copyright © 2019 Joona Havukainen. All rights reserved.
//

#include "Reader_CPU.hpp"
#include <TH1.h>
#include <TChain.h>
#include <TTreeReader.h>
#include <TFile.h>

Reader_CPU::Reader_CPU()
{
    
    //Various numbers of how many objects are needed in the selections
    //Global variables
    globalIndex = 0;
    globalVariables = 3;
    
    //Trigger
    triggerIndex = globalIndex + globalVariables;
    triggerVariables = 3;
    
    //MET Filters
    metFilterIndex = triggerIndex + triggerVariables;
    metFilterVariables = 7;
    
    //Tau
    tauIndex = metFilterIndex + metFilterVariables;
    nTaus = 8;
    variablesPerTau = 11;
    
    //HLT taus for matching
    hltIndex = tauIndex + nTaus * variablesPerTau;
    nHLTTaus = nTaus;
    variablesPerHLTTau = 4;
    
    //Jets
    jetIndex = hltIndex + nHLTTaus * variablesPerHLTTau;
    nJets = 1;
    variablesPerJet = 8;
    
    //MET
    metIndex = jetIndex + nJets * variablesPerJet;
    metVariables = 2;
    
    numberOfVariables = globalVariables + triggerVariables + metFilterVariables + nTaus*variablesPerTau + nHLTTaus*variablesPerHLTTau + nJets*variablesPerJet + metVariables;
    batchSize = 2000;
    //    arrayToGPU = new float[numberOfVariables*batchSize];
    nCores = 1;
    entryIndex = 0;
    
}
Reader_CPU::~Reader_CPU(){};

void Reader_CPU::readToArray()
{
    TChain chain("Events");
    for(std::string & file : inputFiles)
    {
        chain.Add(file.c_str(), -1);
    }
    long long nEntries = chain.GetEntries();
    TTreeReader reader(&chain);
    reader.SetEntriesRange(0, 10000000);
    
    long long batches = nEntries/batchSize;
    
    TFile *f = new TFile("histos.root","recreate");
    TH1F passHist("passed","passed;Events", 2, 0, 2);
    
    //Trigger
    TTreeReaderValue<Double_t> L1MET_x(reader, "L1MET_x");
    TTreeReaderValue<Double_t> L1MET_y(reader, "L1MET_y");
    TTreeReaderValue<Bool_t> HLT_LooseIsoPFTau50_Trk30_eta2p1_MET80_vx(reader, "HLT_LooseIsoPFTau50_Trk30_eta2p1_MET80_vx");
    
    //METFilters
    TTreeReaderValue<Bool_t> METFilter_Flag_HBHENoiseFilter(reader, "METFilter_Flag_HBHENoiseFilter");
    TTreeReaderValue<Bool_t> METFilter_Flag_HBHENoiseIsoFilter(reader, "METFilter_Flag_HBHENoiseIsoFilter");
    TTreeReaderValue<Bool_t> METFilter_Flag_CSCTightHaloFilter(reader, "METFilter_Flag_CSCTightHaloFilter");
    TTreeReaderValue<Bool_t> METFilter_Flag_EcalDeadCellTriggerPrimitiveFilter(reader, "METFilter_Flag_EcalDeadCellTriggerPrimitiveFilter");
    TTreeReaderValue<Bool_t> METFilter_Flag_goodVertices(reader, "METFilter_Flag_goodVertices");
    TTreeReaderValue<Bool_t> METFilter_Flag_eeBadScFilter(reader, "METFilter_Flag_eeBadScFilter");
    TTreeReaderValue<Bool_t> METFilter_Flag_globalTightHalo2016Filter(reader, "METFilter_Flag_globalTightHalo2016Filter");
    //        TTreeReaderValue<Bool_t> METFilter_Flag_hbheNoiseTokenRun2Loose(reader, "METFilter_Flag_hbheNoiseTokenRun2Loose");
    //        TTreeReaderValue<Bool_t> METFilter_Flag_hbheNoiseTokenRun2Tight(reader, "METFilter_Flag_hbheNoiseTokenRun2Tight");
    //        TTreeReaderValue<Bool_t> METFilter_Flag_hbheIsoNoiseToken(reader, "METFilter_Flag_hbheIsoNoiseToken");
    //        TTreeReaderValue<Bool_t> METFilter_Flag_badPFMuonFilter(reader, "METFilter_Flag_badPFMuonFilter");
    //        TTreeReaderValue<Bool_t> METFilter_Flag_badChargedCandidateFilter(reader, "METFilter_Flag_badChargedCandidateFilter");
    
    //Taus
    TTreeReaderValue<std::vector<Double_t>> taus_pt(reader, "Taus_pt");
    TTreeReaderValue<std::vector<Double_t>> taus_eta(reader, "Taus_eta");
    TTreeReaderValue<std::vector<Double_t>> taus_phi(reader, "Taus_phi");
    TTreeReaderValue<std::vector<Double_t>> taus_lChTrkPt(reader, "Taus_lChTrkPt");
    TTreeReaderValue<std::vector<Double_t>> taus_lChTrkEta(reader, "Taus_lChTrkEta");
    TTreeReaderValue<std::vector<Short_t>> taus_nProngs(reader, "Taus_nProngs");
    TTreeReaderValue<std::vector<Short_t>> taus_decayMode(reader, "Taus_decayMode");
    TTreeReaderValue<std::vector<Bool_t>> taus_decayModeFinding(reader, "Taus_decayModeFinding");
    TTreeReaderValue<std::vector<Bool_t>> taus_againstElectron(reader, "Taus_againstElectronTightMVA6");
    TTreeReaderValue<std::vector<Bool_t>> taus_againstMuon(reader, "Taus_againstMuonLoose3");
    TTreeReaderValue<std::vector<Bool_t>> taus_isolation(reader, "Taus_byLooseIsolationMVArun2v1DBoldDMwLT");
    
    //HLT Taus
    TTreeReaderValue<std::vector<Double_t>> HLTTau_pt(reader, "HLTTau_pt");
    TTreeReaderValue<std::vector<Double_t>> HLTTau_eta(reader, "HLTTau_eta");
    TTreeReaderValue<std::vector<Double_t>> HLTTau_phi(reader, "HLTTau_phi");
    TTreeReaderValue<std::vector<Double_t>> HLTTau_e(reader, "HLTTau_e");
    
    //MET
    TTreeReaderValue<Double_t> MET_Type1_x(reader, "MET_Type1_x");
    TTreeReaderValue<Double_t> MET_Type1_y(reader, "MET_Type1_y");
    
    //Jets
    TTreeReaderValue<std::vector<Double_t>> jets_pt(reader, "Jets_pt");
    TTreeReaderValue<std::vector<Double_t>> jets_eta(reader, "Jets_eta");
    TTreeReaderValue<std::vector<Double_t>> jets_phi(reader, "Jets_phi");
    TTreeReaderValue<std::vector<Double_t>> jets_e(reader, "Jets_e");
    TTreeReaderValue<std::vector<Short_t>> jets_pdgId(reader, "Jets_pdgId");
    TTreeReaderValue<std::vector<int>> jets_hadronFlavour(reader, "Jets_hadronFlavour");
    TTreeReaderValue<std::vector<int>> jets_partonFlavour(reader, "Jets_partonFlavour");
    TTreeReaderValue<std::vector<bool>> jets_IDloose(reader, "Jets_IDloose");
    TTreeReaderValue<std::vector<bool>> jets_IDtight(reader, "Jets_IDtight");
    TTreeReaderValue<std::vector<bool>> jets_PUIDloose(reader, "Jets_PUIDloose");
    TTreeReaderValue<std::vector<bool>> jets_PUIDtight(reader, "Jets_PUIDtight");

    float *arrayToGPU = new float[numberOfVariables*batchSize];
    int batchEntries = -1;
    while(reader.Next())
    {
        batchEntries++;
        if(batchEntries == batchSize)
        {
            int passed = 0;
            passed = wrapper(arrayToGPU, batchSize, this->getNumberOfVariables(), this->getTauIndex(), this->getHltIndex(),this->getNTaus());
            std::cout<<"Passed "<<passed<<std::endl;
            
            passHist.Fill(0.0, (float)batchSize);
            passHist.Fill(1.0, (float)passed);


            arrayToGPU = new float[numberOfVariables*batchSize];
            batchEntries = 0;
        }

        int localIndex = 0;
        //Global variables
        arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 0] = std::min((int)taus_pt->size(), nTaus); //nTaus
        arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 1] = std::min((int)HLTTau_pt->size(), nHLTTaus); //nHLTTaus
        arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 2] = std::min((int)jets_pt->size(), nHLTTaus); //nJets
        localIndex += globalVariables;
        //            std::cout<<localIndex<<" ";
        
        
        //Trigger variables
        arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 0] = *L1MET_x;
        arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 1] = *L1MET_y;
        arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 2] = *HLT_LooseIsoPFTau50_Trk30_eta2p1_MET80_vx;
        localIndex += triggerVariables;
        //            std::cout<<localIndex<<" ";
        
        //METFilter variables
        arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 0] = *METFilter_Flag_HBHENoiseFilter;
        arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 1] = *METFilter_Flag_HBHENoiseIsoFilter;
        arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 2] = *METFilter_Flag_CSCTightHaloFilter;
        arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 3] = *METFilter_Flag_EcalDeadCellTriggerPrimitiveFilter;
        arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 4] = *METFilter_Flag_goodVertices;
        arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 5] = *METFilter_Flag_eeBadScFilter;
        arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 6] = *METFilter_Flag_globalTightHalo2016Filter;
        //            arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 7] = *METFilter_Flag_hbheNoiseTokenRun2Loose;
        //            arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 7] = *METFilter_Flag_hbheNoiseTokenRun2Tight;
        //            arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 7] = *METFilter_Flag_hbheIsoNoiseToken;
        //            arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 7] = *METFilter_Flag_badPFMuonFilter;
        //            arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 7] = *METFilter_Flag_badChargedCandidateFilter;
        localIndex += metFilterVariables;
        
        //Tau variables
        for(int j=0; j<nTaus; j++)
        {
            if(taus_pt->size()>j)
            {
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 0] = taus_pt->at(j);
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 1] = taus_eta->at(j);
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 2] = taus_phi->at(j);
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 3] = taus_lChTrkPt->at(j);
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 4] = taus_lChTrkEta->at(j);
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 5] = taus_nProngs->at(j);
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 6] = taus_decayMode->at(j);
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 7] = taus_decayModeFinding->at(j);
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 8] = taus_againstElectron->at(j);
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 9] = taus_againstMuon->at(j);
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 10] = taus_isolation->at(j);
            }
            else
            {
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 0] = -99;
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 1] = -99;
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 2] = -99;
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 3] = -99;
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 4] = -99;
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 5] = -99;
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 6] = -99;
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 7] = -99;
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 8] = -99;
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 9] = -99;
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 10] = -99;
            }
            localIndex += variablesPerTau;
            //                std::cout<<localIndex<<" ";
        }
        //End of tau variables
        
        //HLTTau variables
        for (int j = 0; j<nHLTTaus; j++) {
            if(HLTTau_pt->size()>j)
            {
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 0] = HLTTau_pt->at(j);
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 1] = HLTTau_eta->at(j);
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 2] = HLTTau_phi->at(j);
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 3] = HLTTau_e->at(j);
            }
            else
            {
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 0] = -99;
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 1] = -99;
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 2] = -99;
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 3] = -99;
            }
            localIndex += variablesPerHLTTau;
        }
        
        //Jet variables
        for(int j=0; j<nJets; j++)
        {
            if(jets_pt->size()>j)
            {
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex)+0] = jets_pt->at(j);
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex)+1] = jets_eta->at(j);
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex)+2] = jets_phi->at(j);
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex)+3] = jets_e->at(j);
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex)+4] = jets_IDloose->at(j);
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex)+5] = jets_IDtight->at(j);
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex)+6] = jets_PUIDloose->at(j);
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex)+7] = jets_PUIDtight->at(j);
                
                
            }
            else
            {
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 0] = -99;
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 1] = -99;
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 2] = -99;
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 3] = -99;
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 4] = -99;
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 5] = -99;
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 6] = -99;
                arrayToGPU[(batchEntries*numberOfVariables)+(localIndex) + 7] = -99;
            }
            localIndex += variablesPerJet;
            //                std::cout<<localIndex<<" ";
        }
        
        //MET variables
        arrayToGPU[(batchEntries*numberOfVariables) + (localIndex) + 0] = *MET_Type1_x;
        arrayToGPU[(batchEntries*numberOfVariables) + (localIndex) + 1] = *MET_Type1_y;
        localIndex += metVariables;
        //            std::cout<<localIndex<<" ";
        //            std::cout<<std::endl;
    }
    passHist.Write();
    f->Close();
    
}


void Reader_CPU::setFiles(std::vector<std::string> files)
{
    inputFiles = files;
}

float* Reader_CPU::getArrayToGPU()
{
    return arrayToGPU;
}

int Reader_CPU::getNumberOfVariables()
{
    return numberOfVariables;
}

int Reader_CPU::getBatchSize()
{
    return batchSize;
}

int Reader_CPU::getTauIndex()
{
    return tauIndex;
}

int Reader_CPU::getHltIndex()
{
    return hltIndex;
}

int Reader_CPU::getNTaus()
{
    return nTaus;
}
