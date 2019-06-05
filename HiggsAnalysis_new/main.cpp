//
//  main.cpp
//  HiggsAnalysis_new
//
//  Created by Joona Havukainen on 5/6/19.
//  Copyright © 2019 Joona Havukainen. All rights reserved.
//

#include <iostream>
//#include "Selections/TauSelection.hpp"
//#include "Reader.hpp"
#include "Reader_CPU.hpp"
#include <vector>
#include <boost/filesystem.hpp>
#include <ROOT/TTreeProcessorMT.hxx>
#include "CudaSelections/wrapper.h"

int main(int argc, const char * argv[]) {

//    std::string files[1] = {"/Users/hajohajo/Documents/histograms-TT-0.root"};
//    std::string files[1] = {"/work/data/multicrab_Training/histograms-TT-0.root"};
//    std::string files[1] = {"/work/data/multicrab_Training/histograms-ChargedHiggs_HplusTB_HplusToTauNu_M_500-0.root"};
    
    std::vector<std::string> filevector;

    boost::filesystem::path _path("/work/data/multicrab_Dummy/ChargedHiggs_HplusTB_HplusToTauNu_HeavyMass_M_750/results");
    for(boost::filesystem::directory_iterator itr(_path); itr!=boost::filesystem::directory_iterator();++itr)
    {
        filevector.push_back(itr->path().string());
    }

//    filevector.assign(&files[0], &files[0]+1);
    
    Reader_CPU myReader = Reader_CPU();
    myReader.setFiles(filevector);
    myReader.readToArray();

//    float* myArray = myReader.getArrayToGPU();

/*
    for(int i=0; i<myReader.getBatchSize()*myReader.getNumberOfVariables();i++)
    {
        if(i%myReader.getNumberOfVariables() == 0)
        {
            std::cout<<std::endl<<std::endl;
        }
        std::cout<<myArray[i]<<" ";

    }
*/
//    wrapper(myArray, myReader.getBatchSize(), myReader.getNumberOfVariables(), myReader.getTauIndex(), myReader.getHltIndex(),myReader.getNTaus());
    
}
