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
	float dphi = std::fabs(std::fabs(phi2-phi1)-M_PI);

	return (float)std::sqrt(deta*deta + dphi*dphi);
}

__device__
bool passTriggerMatching(int tauInd, int firstHLTTauInd, int nHLTTaus, float triggerTauMatchingCone, float *inputArray)
{
	float myMinDeltaR = 9999.0;
	for(int i=0; i<nHLTTaus; i++)
	{
		myMinDeltaR = fmin(deltaR(inputArray[tauInd+1], inputArray[firstHLTTauInd+(i*4)+1],inputArray[tauInd+2], inputArray[firstHLTTauInd+(i*4)+2]),myMinDeltaR);
	}

	return myMinDeltaR<triggerTauMatchingCone;

}

__global__
void tauSelection(float *inputArray, bool *passedArray, float *numericalResults, int variablesPerEvent, int tauIndex, int hltIndex, int nTaus)
{
	//Index of the processed event
	int processIndex = blockIdx.x * blockDim.x + threadIdx.x;

	//Index of the first variable of the event processed in the inputArray
	int localIndex = processIndex * variablesPerEvent;

	for(int j=0; j<nTaus; j++)
	{
		passedArray[processIndex*3]=passTriggerMatching(localIndex+tauIndex+j*11, localIndex+hltIndex, nTaus,  1.0, inputArray);
	}


}
