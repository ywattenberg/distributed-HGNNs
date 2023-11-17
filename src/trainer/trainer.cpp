#include <torch/torch.h>
#include <vector>
#include <iostream>
#include "trainer.h"


#include "../model/model.h"
#include "../utils/scores.h"
#include "../utils/configParse.h"


using LossFunction = at::Tensor(*)(const at::Tensor&, const at::Tensor&);


void train_model(const ConfigProperties& config, torch::Tensor &labels, torch::Tensor &input_features, LossFunction loss_fn, Model *model){
    
    double lr = config.trainer_properties.learning_rate;
    int n_epochs = config.trainer_properties.epochs;
    int stepsize_output = config.trainer_properties.output_stepsize;
    long train_set_cutoff = config.data_properties.test_idx;

    int lr_step_size = config.lr_scheduler_properties.step_size;
    double lr_gamma = config.lr_scheduler_properties.gamma;

    torch::Tensor train_labels = labels.index({at::indexing::Slice(0,train_set_cutoff)});
    torch::Tensor test_labels = labels.index({at::indexing::Slice(train_set_cutoff,labels.size(0))});

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(lr).weight_decay(0.0005));

    torch::optim::StepLR lr_scheduler = torch::optim::StepLR(optimizer, lr_step_size, lr_gamma);
    
    for (int epoch = 0; epoch < n_epochs; epoch++){
        torch::Tensor predictions = model->forward(input_features);
        torch::Tensor train_predictions = predictions.index({at::indexing::Slice(0,train_set_cutoff)});
        torch::Tensor loss = loss_fn(train_predictions, train_labels);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        lr_scheduler.step();

        if (epoch % stepsize_output == 0){
            {
                torch::NoGradGuard no_grad;
                torch::Tensor test_predictions = predictions.index({at::indexing::Slice(train_set_cutoff,labels.size(0))});
                torch::Tensor test_loss = loss_fn(test_predictions, test_labels);
                torch::Tensor acc = accuracy(test_predictions, test_labels);
                torch::Tensor f1 = f1_score(test_predictions, test_labels, 40);

                std::cout << "Epoch [" << epoch << "/" << n_epochs << "], ";
                std::cout << "Train Loss: " << loss.item<double>() << ", ";
                std::cout << "Test Loss: " << test_loss.item<double>() << ", ";
                std::cout << "Test Accuracy: " << acc.item<double>() << ", ";
                std::cout << "Test F1: " << f1.item<double>() << std::endl;

                // << ", Predictions: " << round(predictions,2)
            }
            
        }
    }
}

