#include <torch/torch.h>
#include <vector>
#include <iostream>
#include "../model/model.h"

using LossFunction = at::Tensor(*)(const at::Tensor&, const at::Tensor&);

void train_model(int n_epochs, int stepsizeOutput, torch::Tensor &labels, torch::Tensor &input_features, LossFunction loss_fn, Model *model, double lr){
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(lr));
    // torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(lr));

    for (int epoch = 0; epoch < n_epochs; epoch++){
        torch::Tensor predictions = model->forward(input_features);
        torch::Tensor loss = loss_fn(predictions, labels);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if (epoch % stepsizeOutput == 0){
            std::cout << "Epoch [" << epoch << "/" << n_epochs << "], Loss: " << loss.item<double>() 
            // << ", Predictions: " << std::round(predictions)
            << std::endl;
        }
    }
}