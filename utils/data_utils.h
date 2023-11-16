#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>
#include <string>

YAML::Node load_config(std::string path){
    YAML::Node config = YAML::LoadFile(path);
    return config;
}

std::string get_filename(std::string name) {
    std::string filename = name + ".txt";
    return filename;
}

void record_timing(std::string filename, int duration) {
    std::ofstream myfile;
    myfile.open(filename, std::ios_base::app);
    myfile << duration << "\n";
    myfile.close();
}


#endif