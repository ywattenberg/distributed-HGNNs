#ifndef FILE_PARSE_H
#define FILE_PARSE_H

#include <torch/torch.h>
#include <filesystem>
#include <string>
#include <fstream>
#include <tuple>
#include <iostream>
#include <vector>



template<class T>
std::tuple<int,int> csvToArray(const std::string &&filePath, std::vector<T> &parsedCsv) {
    if(!(std::filesystem::exists(filePath) && std::filesystem::is_regular_file(filePath))){
        std::cout << "File " << filePath << " does not exist or is not a regular file" << std::endl;
        exit(1);
    }
    std::fstream data(filePath, std::fstream::in);
    std::string line;
    int lines = 0;
    int illegal_elements = 0;
    while(std::getline(data,line))
    {        
        ++lines;
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<std::string> parsedRow;

        while(std::getline(lineStream,cell,','))
        {
            parsedRow.push_back(cell);

            try {
                if(std::is_same<T, float>::value)
                    parsedCsv.push_back(std::stof(cell));
                else if (std::is_same<T, double>::value)
                    parsedCsv.push_back(std::stod(cell));
                else if (std::is_same<T, int>::value)
                    parsedCsv.push_back(std::stoi(cell));

            } catch(std::invalid_argument&) {
                // non parsable row
                std::cout << "Element in row was not parseable" << std::endl;
                while (std::getline(lineStream,cell,','));
                lines--;
            }
        }
    }

    return std::make_tuple(lines, (parsedCsv.size())/lines);
}


template<class T>
inline torch::Tensor tensor_from_file(const std::string& path){
  std::vector<T> data;
  auto [lines, cols] = csvToArray(std::move(path), data);
  return torch::from_blob(data.data(), {lines,cols}).clone();
}
#endif
