#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <yaml-cpp/yaml.h>

#include "model/model.h"
#include "trainer/trainer.h"
#include "utils/data_utils.h"

using LossFunction = at::Tensor(*)(const at::Tensor&, const at::Tensor&); //Supertype for loss functions


int main(){
  // load config
  std::string config_path = std::string(std::filesystem::current_path()) + "/../experiments/test.yaml";
  YAML::Node config = load_config(config_path);
  std::cout << "Config: " << config << std::endl;

  // PARAMETERS
  YAML::Node model_conf = config["model"];
  YAML::Node trainer_conf = config["trainer"];
  int DATA_SAMPLES = model_conf["data_samples"].as<int>();
  int FEATURE_DIMENSIONS =  model_conf["feature_dimensions"].as<int>();
  std::vector<int> HIDDEN_DIMS =  model_conf["hidden_dims"].as<std::vector<int>>();
  double DROPOUT =  model_conf["dropout_rate"].as<double>();
  bool WITH_BIAS =  model_conf["with_bias"].as<bool>();

  double LEARNING_RATE = trainer_conf["learning_rate"].as<double>();
  int EPOCHS = trainer_conf["epochs"].as<int>();
  int OUTPUT_STEPSIZE = trainer_conf["output_stepsize"].as<int>(); //interval of epochs to output the loss
  
  int CLASSES = 10; //config["classes"].as<int>();

  // Load Data - Currently done randomly
  torch::Tensor features = torch::rand({DATA_SAMPLES, FEATURE_DIMENSIONS});
  torch::Tensor labels = torch::randint(0, CLASSES-1, {DATA_SAMPLES});
  torch::Tensor leftSide = torch::eye(DATA_SAMPLES);

  std::cout << "Labels: " << std::endl;
  labels.print();


  // Build Model
  auto model = new Model(FEATURE_DIMENSIONS, HIDDEN_DIMS, CLASSES, DROPOUT, &leftSide, WITH_BIAS);
  std::cout << "Model parameters before training: " << std::endl;
  for (const auto& params :  model->parameters()) {
    float* buffer = params.data_ptr<float>();
    std::cout<<buffer[0]<<std::endl;
  }

  // Define the loss function
  LossFunction ce_loss_fn = [](const torch::Tensor& predicted, const torch::Tensor& target) {
        return torch::nn::functional::cross_entropy(predicted, target);
    };
    

  // Train the model
  model->train();
  train_model(EPOCHS, OUTPUT_STEPSIZE, labels, features, ce_loss_fn, model, LEARNING_RATE);

  std::cout << "Model parameters after training: "  << std::endl;
 for (const auto& params :  model->parameters()) {
    float* buffer = params.data_ptr<float>();
    std::cout<<buffer[0]<<std::endl;
  }

}

