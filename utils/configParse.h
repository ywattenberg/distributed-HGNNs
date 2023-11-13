#ifndef CONFIG_PARSE_H
#define CONFIG_PARSE_H

#include <yaml-cpp/yaml.h>
#include <iostream>

//Struct for config parameters
struct ConfigProperties
{   
    double dropout_rate;
    double learning_rate;
    bool with_bias;
    int epochs;
    int output_stepsize; 
    int classes;
    long test_idx;
    
    std::string g_path;
    std::string labels_path;
    std::string features_path;
    std::vector<int> hidden_dims;
};

//Function to parse config file
ConfigProperties ParseConfig(const std::string& config_path);

#endif