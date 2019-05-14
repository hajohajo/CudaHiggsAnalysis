//
//  main.cpp
//  HiggsAnalysis_new
//
//  Created by Joona Havukainen on 5/6/19.
//  Copyright © 2019 Joona Havukainen. All rights reserved.
//

#include <iostream>
#include "Selections/TauSelection.hpp"
#include "Reader.hpp"
#include <vector>
#include <ROOT/TTreeProcessorMT.hxx>

int main(int argc, const char * argv[]) {

    std::string files[1] = {"/Users/hajohajo/Documents/histograms-TT-0.root"};

    std::vector<std::string> filevector;
    filevector.assign(&files[0], &files[0]+1);
    
    Reader myReader = Reader();
    myReader.setFiles(filevector);
    myReader.readToArray();

    float* myArray = myReader.getArrayToGPU();

    for(int i=0; i<myReader.getBatchSize()*myReader.getNumberOfVariables();i++)
    {
        if(i%myReader.getNumberOfVariables() == 0)
        {
            std::cout<<std::endl<<std::endl;
        }
        std::cout<<myArray[i]<<" ";

    }
    
}
