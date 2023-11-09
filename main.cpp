#include <iostream>
#include <filesystem>
#include <unistd.h>

#include <yaml-cpp/yaml.h>
#include <torch/torch.h>

#include "model/model.h"
#include "trainer/trainer.h"
#include "utils/fileParse.h"
#include "utils/configParse.h"



using LossFunction = at::Tensor(*)(const at::Tensor&, const at::Tensor&); //Supertype for loss functions


int main(int argc, char** argv){
  // read command line arguments
  int opt;
  std::string config_path;
  while((opt = getopt(argc, argv, "c:")) != -1){
    switch(opt){
      case 'c':
        config_path = optarg;
        break;
      default:
        std::cerr << "Usage: " << argv[0] << " -c <config_path>" << std::endl;
        exit(EXIT_FAILURE);
    }
  }

  // load config
  ConfigProperties config = ParseConfig(config_path);

  std::vector<float> data;
  auto [G_lines, G_cols] = csvToArray(std::move(config.G_path), data);
  torch::Tensor leftSide = torch::from_blob(data.data(), {G_lines,G_cols});
  std::cout << "G dimensions: " << G_lines << "x" << G_cols << std::endl;
  
  // Convert the leftSide tensor to a sparse tensor
  torch::Tensor index = leftSide.index({at::indexing::Slice(), at::indexing::Slice(0,2)}).transpose(0,1);
  index = index.to(torch::kLong);
  torch::Tensor values = leftSide.index({at::indexing::Slice(), at::indexing::Slice(2,3)}).squeeze();
  // This way of calling sparse_coo_tensor assumes 
  // that we have at least one non-zero value in each row/column 
  leftSide = torch::sparse_coo_tensor(index, values);
  std::cout << "leftSide dimensions: " << leftSide.sizes() << std::endl;

  std::vector<float> data2;
  auto [L_lines, L_cols] = csvToArray(std::move(config.labels_path), data2);
  torch::Tensor labels = torch::from_blob(data2.data(), {L_lines,L_cols});
  labels = labels.index({at::indexing::Slice(), at::indexing::Slice(1)}).squeeze().to(torch::kLong);
  std::cout << "labels shape: " << labels.sizes() << std::endl;
 
  std::vector<int> data3; 
  auto [F_lines, F_cols] = csvToArray(std::move(config.features_path), data3);
  torch::Tensor features = torch::from_blob(data3.data(), {F_lines,F_cols});
  std::cout << "features shape: " << features.sizes() << std::endl;

  // Build Model
  auto model = new Model(F_cols, config.hidden_dims, config.classes, config.dropout_rate, &leftSide, config.with_bias);

  // Define the loss function
  LossFunction ce_loss_fn = [](const torch::Tensor& predicted, const torch::Tensor& target) {
        return torch::nn::functional::cross_entropy(predicted, target);
    };
  // Train the model
  train_model(config.epochs, config.output_stepsize, labels, features, ce_loss_fn, model, config.learning_rate);
}
