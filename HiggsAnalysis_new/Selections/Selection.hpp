//
//  Selection.hpp
//  HiggsAnalysis_new
//
//  Created by Joona Havukainen on 5/7/19.
//  Copyright © 2019 Joona Havukainen. All rights reserved.
//

#ifndef Selection_hpp
#define Selection_hpp

#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <ROOT/TTreeProcessorMT.hxx>

//Convenience structure to store information of variables to be used in the selection
class Selection
{
private:
    uint nVariables;
    uint startIndex;
    std::vector<std::string> *variableNames;
    std::vector<uint> *variableIndices;
    TTreeReader *reader;
public:
    uint getNvariables();
    TTreeReader* getReader();
    void setReader(TTreeReader* reader);
    std::vector<uint> *getVariablesIndices();
    std::vector<std::string> *getVariableNames();
    void setNvariables(uint number);
    void setVariableIndices(std::vector<uint> indices);
    void setVariableNames(std::vector<std::string> names);
    bool makeSelection();
    Selection(uint currentIndex);
    ~Selection();
};


#endif /* Selection_hpp */
