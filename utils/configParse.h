#ifndef CONFIG_PARSE_H
#define CONFIG_PARSE_H

#include <yaml-cpp/yaml.h>
#include <iostream>

// Struct for model parameters
struct ModelProperties
{
    int classes;
    std::vector<int> hidden_dims;
    double dropout_rate;
    bool with_bias;
    std::string activation;
};

// Struct for trainer parameters
struct TrainerProperties
{
    double learning_rate;
    int epochs;
    int output_stepsize;
    std::string loss_function;
};

// Struct for lr_scheduler parameters
struct LRSchedulerProperties
{
    int step_size;
    double gamma;
};

// Struct for data parameters
struct DataProperties
{
    std::string g_path;
    std::string labels_path;
    std::string features_path;
    long test_idx;
};

//Struct for config parameters
struct ConfigProperties
{   
    ModelProperties model_properties;
    TrainerProperties trainer_properties;
    LRSchedulerProperties lr_scheduler_properties;
    std::string task_type;
    DataProperties data_properties;
};

//Function to parse config file
ConfigProperties ParseConfig(const std::string& config_path);

#endif