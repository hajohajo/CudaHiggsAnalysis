//
//  TriggerSelection.hpp
//  HiggsAnalysis_new
//
//  Created by Joona Havukainen on 5/31/19.
//  Copyright © 2019 Joona Havukainen. All rights reserved.
//

#pragma once

extern __global__ void triggerSelection(float *inputArray, bool *passedArray, int variablesPerEvent, int nEvents, int triggerIndex);
