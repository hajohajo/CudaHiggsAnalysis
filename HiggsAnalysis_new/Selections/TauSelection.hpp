//
//  TauSelection.hpp
//  HiggsAnalysis_new
//
//  Created by Joona Havukainen on 5/8/19.
//  Copyright © 2019 Joona Havukainen. All rights reserved.
//

#ifndef TauSelection_hpp
#define TauSelection_hpp

#include <stdio.h>
#include <vector>
#include "Selection.hpp"

class TauSelection: public Selection
{
public:
    bool makeSelection();
    TauSelection(uint currentIndex);
    ~TauSelection();
};

#endif /* TauSelection_hpp */
