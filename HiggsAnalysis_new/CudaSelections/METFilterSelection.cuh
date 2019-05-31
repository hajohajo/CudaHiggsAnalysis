//
//  METFilterSelection.hpp
//  HiggsAnalysis_new
//
//  Created by Joona Havukainen on 5/31/19.
//  Copyright © 2019 Joona Havukainen. All rights reserved.
//

#pragma once

extern __global__ void metFilterSelection(float *inputArray, bool *passedArray, bool *passed, int variablesPerEvent, int nEvents, int metFilterIndex);
