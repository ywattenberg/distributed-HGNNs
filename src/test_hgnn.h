#pragma once

#include "SpmatLocal.hpp"
#include <string>


void benchmark_algorithm(SpmatLocal* S, 
        string algorithm_name,
        string output_file,
        bool fused,
        int R,
        int c,
        string app 
        );