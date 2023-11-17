#include "../utils/configParse.h"
#include <yaml-cpp/yaml.h>
#include <iostream>

//Function to parse config file
ConfigProperties ParseConfig(const std::string& config_path){
    YAML::Node config = YAML::LoadFile(config_path);
    std::cout << "Using config in path: " << config_path << std::endl;

    ConfigProperties config_properties;

    // model properties
    config_properties.model_properties.classes = config["model"]["classes"].as<int>();
    config_properties.model_properties.hidden_dims = config["model"]["hidden_dims"].as<std::vector<int>>();
    config_properties.model_properties.dropout_rate = config["model"]["dropout_rate"].as<double>();
    config_properties.model_properties.with_bias = config["model"]["with_bias"].as<bool>();
    config_properties.model_properties.activation = config["model"]["activation"].as<std::string>();

    // trainer properties
    config_properties.trainer_properties.learning_rate = config["trainer"]["learning_rate"].as<double>();
    config_properties.trainer_properties.epochs = config["trainer"]["epochs"].as<int>();
    config_properties.trainer_properties.output_stepsize = config["trainer"]["output_stepsize"].as<int>();
    config_properties.trainer_properties.loss_function = config["trainer"]["loss_function"].as<std::string>();

    // lr_scheduler properties
    config_properties.lr_scheduler_properties.step_size = config["lr_scheduler"]["step_size"].as<int>();
    config_properties.lr_scheduler_properties.gamma = config["lr_scheduler"]["gamma"].as<double>();

    // task type
    config_properties.task_type = config["task_type"].as<std::string>();

    // data properties
    config_properties.data_properties.g_path = config["data"]["g_path"].as<std::string>();
    config_properties.data_properties.labels_path = config["data"]["labels_path"].as<std::string>();
    config_properties.data_properties.features_path = config["data"]["features_path"].as<std::string>();
    config_properties.data_properties.test_idx = config["data"]["test_idx"].as<long>();
    
    return config_properties;
}