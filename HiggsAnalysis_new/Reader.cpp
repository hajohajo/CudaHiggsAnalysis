//
//  Reader.cpp
//  HiggsAnalysis_new
//
//  Created by Joona Havukainen on 5/13/19.
//  Copyright © 2019 Joona Havukainen. All rights reserved.
//

#include "Reader.hpp"

Reader::Reader()
{
    
    //Various numbers of how many objects are needed in the selections

    //Global variables
    globalIndex = 0;
	globalVariables = 3;
    
    //Trigger
    triggerIndex = globalIndex + globalVariables;
    triggerVariables = 3;
    
    //Tau
    tauIndex = triggerIndex + triggerVariables;
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

    numberOfVariables = nTaus*variablesPerTau + nJets*variablesPerJet + metVariables;
    batchSize = 10;
    arrayToGPU = new float[numberOfVariables*batchSize];
    nCores = 4;
    entryIndex = 0;
    
}
Reader::~Reader(){};

void Reader::readToArray()
{
    std::vector<std::string_view> files_stringview;
    for(std::string & file : inputFiles)
    {
        files_stringview.push_back((std::string_view)file.c_str());
    }
    
    ROOT::EnableImplicitMT(nCores);
    ROOT::TTreeProcessorMT treeProcessor(files_stringview, "Events");
    
    auto workItem = [&](TTreeReader &reader)
    {
        reader.SetEntriesRange(entryIndex, entryIndex+batchSize);

        //Trigger
        TTreeReaderValue<Double_t> L1MET_x(reader, "L1MET_x");
        TTreeReaderValue<Double_t> L1MET_y(reader, "L1MET_y");
        TTreeReaderValue<Bool_t> HLT_LooseIsoPFTau50_Trk30_eta2p1_MET80_vx(reader, "HLT_LooseIsoPFTau50_Trk30_eta2p1_MET80_vx");
        
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

        while (reader.Next())
        {
            unsigned long event = reader.GetCurrentEntry()-entryIndex;
//            std::cout<<event<<std::endl;

            int localIndex = 0;
            
			//Global variables
			arrayToGPU[(event*numberOfVariables)+(localIndex) + 0] = std::min((int)taus_pt->size(), nTaus); //nTaus
            arrayToGPU[(event*numberOfVariables)+(localIndex) + 1] = std::min((int)HLTTau_pt->size(), nHLTTaus); //nHLTTaus
            arrayToGPU[(event*numberOfVariables)+(localIndex) + 2] = std::min((int)jets_pt->size(), nHLTTaus); //nJets
			localIndex += globalVariables;


            //Trigger variables
            arrayToGPU[(event*numberOfVariables)+(localIndex) + 0] = *L1MET_x;
            arrayToGPU[(event*numberOfVariables)+(localIndex) + 1] = *L1MET_y;
            arrayToGPU[(event*numberOfVariables)+(localIndex) + 2] = *HLT_LooseIsoPFTau50_Trk30_eta2p1_MET80_vx;
            localIndex+=triggerVariables;
            
            //Tau variables
            for(int j=0; j<nTaus; j++)
            {
                if(taus_pt->size()>j)
                {
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 0] = taus_pt->at(j);
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 1] = taus_eta->at(j);
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 2] = taus_phi->at(j);
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 3] = taus_lChTrkPt->at(j);
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 4] = taus_lChTrkEta->at(j);
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 5] = taus_nProngs->at(j);
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 6] = taus_decayMode->at(j);
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 7] = taus_decayModeFinding->at(j);
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 8] = taus_againstElectron->at(j);
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 9] = taus_againstMuon->at(j);
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 10] = taus_isolation->at(j);
                }
                else
                {
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 0] = -99;
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 1] = -99;
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 2] = -99;
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 3] = -99;
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 4] = -99;
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 5] = -99;
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 6] = -99;
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 7] = -99;
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 8] = -99;
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 9] = -99;
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 10] = -99;
                }
                localIndex=+variablesPerTau;
            }
            //End of tau variables
            
            //HLTTau variables
            for (int j = 0; j<nHLTTaus; j++) {
                if(HLTTau_pt->size()>j)
                {
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 0] = HLTTau_pt->at(j);
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 1] = HLTTau_eta->at(j);
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 2] = HLTTau_phi->at(j);
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 3] = HLTTau_e->at(j);
                }
                else
                {
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 0] = -99;
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 1] = -99;
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 2] = -99;
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 3] = -99;
                }
                localIndex=+variablesPerHLTTau;
            }
            
            //Jet variables
            for(int j=0; j<nJets; j++)
            {
                if(jets_pt->size()>j)
                {
                    arrayToGPU[(event*numberOfVariables)+(localIndex)+0] = jets_pt->at(j);
                    arrayToGPU[(event*numberOfVariables)+(localIndex)+1] = jets_eta->at(j);
                    arrayToGPU[(event*numberOfVariables)+(localIndex)+2] = jets_phi->at(j);
                    arrayToGPU[(event*numberOfVariables)+(localIndex)+3] = jets_e->at(j);
                    arrayToGPU[(event*numberOfVariables)+(localIndex)+4] = jets_IDloose->at(j);
                    arrayToGPU[(event*numberOfVariables)+(localIndex)+5] = jets_IDtight->at(j);
                    arrayToGPU[(event*numberOfVariables)+(localIndex)+6] = jets_PUIDloose->at(j);
                    arrayToGPU[(event*numberOfVariables)+(localIndex)+7] = jets_PUIDtight->at(j);


                }
                else
                {
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 0] = -99;
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 1] = -99;
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 2] = -99;
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 3] = -99;
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 4] = -99;
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 5] = -99;
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 6] = -99;
                    arrayToGPU[(event*numberOfVariables)+(localIndex) + 7] = -99;
                }
                localIndex+=variablesPerJet;
            }
            
            //MET variables
            arrayToGPU[(event*numberOfVariables) + (localIndex) + 0] = *MET_Type1_x;
            arrayToGPU[(event*numberOfVariables) + (localIndex) + 1] = *MET_Type1_y;
            localIndex+=metVariables;
        }
    };
    
    treeProcessor.Process(workItem);
    entryIndex = entryIndex+batchSize;
    std::cout<<"Data read to array"<<std::endl;
    std::cout<<"Entry index: "<<entryIndex<<std::endl;
}

void Reader::setFiles(std::vector<std::string> files)
{
    inputFiles = files;
}

float* Reader::getArrayToGPU()
{
    return arrayToGPU;
}

int Reader::getNumberOfVariables()
{
    return numberOfVariables;
}

int Reader::getBatchSize()
{
    return batchSize;
}

int Reader::getTauIndex()
{
    return tauIndex;
}

int Reader::getHltIndex()
{
    return hltIndex;
}

int Reader:: getNTaus()
{
    return nTaus;
}
