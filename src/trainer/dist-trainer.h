#ifndef DIST_TRAINER_H
#define DIST_TRAINER_H

#include <torch/torch.h>
#include <vector>
#include <iostream>
#include "../model/dist-model.h"
#include "../utils/configParse.h"

void train_dist_model(const ConfigProperties& config, vector<int> &labels, DenseMatrix<double> &input_features, DistModel *model, std::string run_id, bool timing, std::string timing_file);

#endif