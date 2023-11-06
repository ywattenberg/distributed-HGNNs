#ifndef TRAINER_H
#define TRAINER_H

#include <torch/torch.h>
#include <vector>
#include <iostream>
#include "../model/model.h"

using LossFunction = at::Tensor(*)(const at::Tensor&, const at::Tensor&); //Supertype for loss functions


void train_model(int n_epochs, int stepsizeOutput, torch::Tensor &labels, torch::Tensor &input_features, LossFunction loss_fn, Model *model, double lr);

#endif