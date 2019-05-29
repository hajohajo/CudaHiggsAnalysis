//
//  TauSelection.cpp
//  HiggsAnalysis_new
//
//  Created by Joona Havukainen on 5/14/19.
//  Copyright © 2019 Joona Havukainen. All rights reserved.
//

//#include "TauSelection.cuh"

__device__
float deltaR(float eta1, float eta2, float phi1, float phi2)
{
	float deta = eta2-eta1;
	float dphi = phi2-phi1;
	if(dphi>=M_PI)
	{
	    dphi=dphi-2*M_PI;
	}else if(dphi<-M_PI)
	{
	    dphi=dphi+2*M_PI;
	}

	return std::sqrt(deta*deta + dphi*dphi);
}

//Check that tau matches HLT tau
__device__
bool passTriggerMatching(int tauInd, int firstHLTTauInd, int nHLTTaus, float triggerTauMatchingCone, float *inputArray, float *numericalArray, int processIndex)
{
	float myMinDeltaR = 9999.0;
	for(int i=0; i<nHLTTaus; i++)
	{
		myMinDeltaR = fminf(deltaR(inputArray[tauInd+1], inputArray[firstHLTTauInd+(i*4)+1],inputArray[tauInd+2], inputArray[firstHLTTauInd+(i*4)+2]),myMinDeltaR);
	}

	return myMinDeltaR<triggerTauMatchingCone;
}

//Check tau prongs
//Useless at the moment, just accept any number of prongs
__device__
bool passNProngsCut(int tauNProngs, int tauDecayMode)
{
    return true;
}

//Just the bool in tauDecayModeFinding, made into function for consistency
__device__
bool passDecayModeFinding(int tauDecayModeFinding)
{
    return tauDecayModeFinding;
}

//Find out what are generic discriminators
__device__
bool passGenericDiscriminators()
{
    return true;
}

__device__
bool passElectronDiscriminator(int tauElectronDiscriminator)
{
    return tauElectronDiscriminator;
}

__device__
bool passMuonDiscriminator(int tauMuonDiscriminator)
{
    return tauMuonDiscriminator;
}

__device__
bool passTauIsolation(int tauIsolationDiscriminator)
{
    return tauIsolationDiscriminator;
}

__global__
void tauSelection(float *inputArray, bool *passedArray, bool *selectedTaus, float *numericalResults, int variablesPerEvent, int tauIndex, int hltIndex, int nTaus)
{
	//Index of the processed event
	int processIndex = blockIdx.x * blockDim.x + threadIdx.x;

	//Index of the first variable of the event processed in the inputArray
	int localIndex = processIndex * variablesPerEvent;

    //Tau loop
	for(int j=0; j<inputArray[processIndex*variablesPerEvent+0]; j++)
	{
	    int thisTau = processIndex*nTaus+j;
		selectedTaus[thisTau]=passTriggerMatching(localIndex+tauIndex+j*11, localIndex+hltIndex, nTaus,  0.4, inputArray, numericalResults, processIndex);
		selectedTaus[thisTau]=passNProngsCut(inputArray[localIndex+tauIndex+j*11+5], inputArray[localIndex+tauIndex+j*11+6])&&selectedTaus[thisTau];
		selectedTaus[thisTau]=passDecayModeFinding(inputArray[localIndex+tauIndex+j*11+7])&&selectedTaus[thisTau];
		selectedTaus[thisTau]=passGenericDiscriminators()&&selectedTaus[thisTau];
		selectedTaus[thisTau]=passElectronDiscriminator(inputArray[localIndex+tauIndex+j*11+8])&&selectedTaus[thisTau];
		selectedTaus[thisTau]=passMuonDiscriminator(inputArray[localIndex+tauIndex+j*11+9])&&selectedTaus[thisTau];
	    selectedTaus[thisTau]=passTauIsolation(inputArray[localIndex+tauIndex+j*11+10])&&selectedTaus[thisTau];
	}


}
