#ifndef SCORES_H
#define SCORES_H

#include <iostream>
#include <torch/torch.h>

torch::Tensor accuracy(torch::Tensor &y_pred, torch::Tensor &y_true);
torch::Tensor precision(torch::Tensor &y_pred, torch::Tensor &y_true, int label);
torch::Tensor recall(torch::Tensor &y_pred, torch::Tensor &y_true, int label);
torch::Tensor f1_score(torch::Tensor &y_pred, torch::Tensor &y_true, int num_classes);

#endif 
