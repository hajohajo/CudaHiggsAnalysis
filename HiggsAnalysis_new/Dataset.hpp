//
//  Dataset.hpp
//  HiggsAnalysis_new
//
//  Created by Joona Havukainen on 5/6/19.
//  Copyright © 2019 Joona Havukainen. All rights reserved.
//

#ifndef Dataset_hpp
#define Dataset_hpp

#include <stdio.h>
#include <string>

class Dataset
{
private:
    std::string name;
    std::string dataVersion;
    std::string lumiFile;
    int nAllEvents;

public:
    std::string getName();
//    std::string getFileNames();
//    std::string getDataVersion();
//    std::string getLumiFile();
//    std::string getNAllEvents();
//    bool isMC();
};

#endif /* Dataset_hpp */
