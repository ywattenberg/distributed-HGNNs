#ifndef FILE_PARSE_H
#define FILE_PARSE_H

#include <torch/torch.h>
#include <string>
#include <fstream>
#include <tuple>
#include <iostream>
#include <vector>

template<class T>
std::tuple<int,int> csvToArray(std::string &&filePath, std::vector<T> &parsedCsv) {
    std::fstream data(filePath, std::fstream::in);
    std::string line;
    int lines = 0;
    while(std::getline(data,line))
    {
        ++lines;
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<std::string> parsedRow;
        while(std::getline(lineStream,cell,','))
        {
	    if(std::is_same<T, float>::value)
              parsedCsv.push_back(stof(cell));
	    else if (std::is_same<T, double>::value)
              parsedCsv.push_back(stod(cell));
	    else if (std::is_same<T, int>::value)
              parsedCsv.push_back(stoi(cell));
        }
    }
    return std::make_tuple(lines, parsedCsv.size()/lines);
}
#endif
