//
//  TauSelection.cpp
//  HiggsAnalysis_new
//
//  Created by Joona Havukainen on 5/14/19.
//  Copyright © 2019 Joona Havukainen. All rights reserved.
//

#include "TauSelection.cuh"

__device__
float deltaR(float eta1, float eta2, float phi1, float phi2)
{
	float deta = eta2-eta1;
	float dphi = std::fabs(std::fabs(phi2-phi1)-M_PI);

	return (float)std::sqrt(deta*deta + dphi*dphi);
}

__device__
bool passTriggerMatching(int tauInd, int firstHLTTauInd, int nHLTTaus, triggerTauMatchingCone)
{
	float myMinDeltaR = 9999.0;
	for(int i=0; i<nHLTTaus; i++)
	{
		myMinDeltaR = std::min(deltaR(array[tauInd+1], array[firstHLTTauInd+(i*4)+1)],array[tauInd+2], array[firstHLTTauInd+(i*4)+2)]), myMinDeltaR);
	}

	return myMinDeltaR<triggerTauMatchingCone;

}

__global__
void tauSelection(float *array, int variablesPerEvent)
{
	//Index of the processed event
	int processIndex = blockIdx.x * blockDim.x + threadIdx.x;

	//Index of the first variable of the event processed in the array
	int localIndex = processIndex * variablesPerEvent;

	for(int j=0; j<array[localIndex]; j++)
	{
		passTriggerMatchin(localIndex+j*11)
	}


}
