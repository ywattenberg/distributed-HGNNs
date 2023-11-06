#include <yaml-cpp/yaml.h>
#include <iostream>

YAML::Node load_config(std::string path){
    YAML::Node config = YAML::LoadFile(path);
    return config;
}


    