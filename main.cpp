#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "model/model.h"
#include "trainer/trainer.h"

// MODEL PARAMETERS
int DATA_SAMPLES = 10;
int FEATURE_DIMENSIONS = 10;
std::vector<int> HIDDEN_DIMS = {20,10};
double DROPOUT = 0.2;
bool WITH_BIAS = false;
double LEARNING_RATE = 0.001;
int CLASSES = 5;
int EPOCHS = 1000;
int OUTPUT_STEPSIZE = 100; //interval of epochs to output the loss

using LossFunction = at::Tensor(*)(const at::Tensor&, const at::Tensor&); //Supertype for loss functions


int main(){
  
  // Load Data - Currently done randomly
  torch::Tensor features = torch::rand({DATA_SAMPLES, FEATURE_DIMENSIONS});
  torch::Tensor labels = torch::randint(0, CLASSES-1, {DATA_SAMPLES});
  torch::Tensor leftSide = torch::eye(DATA_SAMPLES);

  std::cout << "Labels: " << labels << std::endl;


  // Build Model
  auto model = new Model(FEATURE_DIMENSIONS, HIDDEN_DIMS, CLASSES, DROPOUT, &leftSide, WITH_BIAS);
  // std::cout << "Model parameters before training: " << model->parameters() << std::endl;

  // Define the loss function
  LossFunction ce_loss_fn = [](const torch::Tensor& predicted, const torch::Tensor& target) {
        return torch::nn::functional::cross_entropy(predicted, target);
    };

  // Train the model
  train_model(EPOCHS, OUTPUT_STEPSIZE, labels, features, ce_loss_fn, model, LEARNING_RATE);

  // std::cout << "Model parameters after training: " << model->parameters() << std::endl;

}