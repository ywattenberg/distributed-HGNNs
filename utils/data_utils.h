#ifndef DATA_UTILS
#define DATA_UTILS

#include <yaml-cpp/yaml.h>
#include <iostream>

YAML::Node load_config(std::string path);

auto ConvertYAMLToType(const YAML::Node& node);


#endif