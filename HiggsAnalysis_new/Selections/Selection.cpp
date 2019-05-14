//
//  Selection.cpp
//  HiggsAnalysis_new
//
//  Created by Joona Havukainen on 5/7/19.
//  Copyright © 2019 Joona Havukainen. All rights reserved.
//

#include "Selection.hpp"

Selection::Selection(uint currentIndex)
{
    startIndex = currentIndex;
};
Selection::~Selection(){};

uint Selection::getNvariables()
{
    return nVariables;
}

std::vector<uint>* Selection::getVariablesIndices()
{
    return variableIndices;
}

std::vector<std::string>* Selection::getVariableNames()
{
    return variableNames;
}

void Selection::setNvariables(uint number)
{
    nVariables = number;
}

void Selection::setVariableNames(std::vector<std::string> names)
{
    variableNames = &names;
}

void Selection::setVariableIndices(std::vector<uint> indices)
{
    variableIndices = &indices;
}

bool Selection::makeSelection()
{
    std::cout<<"Override this method in subclasses"<<std::endl;
    return false;
}

TTreeReader* Selection::getReader()
{
    return reader;
}

void Selection::setReader(TTreeReader* reader)
{
    this->reader = reader;
}
