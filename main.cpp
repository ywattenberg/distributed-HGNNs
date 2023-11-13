#include <iostream>
#include <filesystem>
#include <unistd.h>

#include <yaml-cpp/yaml.h>
#include <torch/torch.h>

#include "model/model.h"
#include "trainer/trainer.h"
#include "utils/fileParse.h"
#include "utils/configParse.h"

inline torch::Tensor coo_tensor_to_sparse(torch::Tensor& coo_tensor){
  torch::Tensor index = coo_tensor.index({at::indexing::Slice(), at::indexing::Slice(0,2)}).transpose(0,1);
  index = index.to(torch::kLong);
  torch::Tensor values = coo_tensor.index({at::indexing::Slice(), at::indexing::Slice(2,3)}).squeeze();
  // This way of calling sparse_coo_tensor assumes 
  // that we have at least one non-zero value in each row/column 
  return torch::sparse_coo_tensor(index, values);
}

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

  torch::Tensor coo_list = tensor_from_file<float>(config.g_path);
  torch::Tensor left_side = coo_tensor_to_sparse(coo_list);
  std::cout << "G dimensions: " << left_side.sizes() << std::endl;

  torch::Tensor labels = tensor_from_file<float>(config.labels_path);
  labels = labels.index({at::indexing::Slice(), at::indexing::Slice(1)}).squeeze().to(torch::kLong);
  std::cout << "labels shape: " << labels.sizes() << std::endl;
   
  torch::Tensor features = tensor_from_file<float>(config.features_path);
  std::cout << "features shape: " << features.sizes() << std::endl;
  int f_cols = features.size(1);
  // Build Model
  auto model = new Model(f_cols, config.hidden_dims, config.classes, config.dropout_rate, &left_side, config.with_bias);

  // Define the loss function
  LossFunction ce_loss_fn = [](const torch::Tensor& predicted, const torch::Tensor& target) {
        return torch::nn::functional::cross_entropy(predicted, target);
    };
  // Train the model
  train_model(config, labels, features, ce_loss_fn, model);
}
