#include <torch/torch.h>
#include <vector>
#include <iostream>
#include "../model/model.h"
#include "../utils/configParse.h"


using LossFunction = at::Tensor(*)(const at::Tensor&, const at::Tensor&);


void train_model(const ConfigProperties& config, torch::Tensor &labels, torch::Tensor &input_features, LossFunction loss_fn, Model *model){
    
    double lr = config.learning_rate;
    int n_epochs = config.epochs;
    int stepsize_output = config.output_stepsize;
    long train_set_cutoff = config.test_idx;

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(lr));
    // torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(lr));

    for (int epoch = 0; epoch < n_epochs; epoch++){
        torch::Tensor predictions = model->forward(input_features);
        torch::Tensor loss = loss_fn(predictions, labels);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if (epoch % stepsize_output == 0){
            std::cout << "Epoch [" << epoch << "/" << n_epochs << "], Loss: " << loss.item<double>() 
            // << ", Predictions: " << round(predictions,2)
            << std::endl;
        }
    }
}

