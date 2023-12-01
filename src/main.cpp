#include <iostream>
#include <filesystem>
#include <unistd.h>

#include <yaml-cpp/yaml.h>
#include <torch/torch.h>

#include "model/model.h"
#include "model/dist-model.h"
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

int model(ConfigProperties& config){

  torch::Tensor coo_list = tensor_from_file<float>(config.data_properties.g_path);
  torch::Tensor left_side = coo_tensor_to_sparse(coo_list);
  std::cout << "G dimensions: " << left_side.sizes() << std::endl;

  torch::Tensor labels = tensor_from_file<float>(config.data_properties.labels_path);
  labels = labels.index({at::indexing::Slice(), at::indexing::Slice(-1)}).squeeze().to(torch::kLong);
  std::cout << "labels shape: " << labels.sizes() << std::endl;
   
  torch::Tensor features = tensor_from_file<float>(config.data_properties.features_path);
  std::cout << "features shape: " << features.sizes() << std::endl;
  int f_cols = features.size(1);
  // Build Model
  auto model = new Model(f_cols, config.model_properties.hidden_dims, config.model_properties.classes, config.model_properties.dropout_rate, &left_side, config.model_properties.with_bias);
  // Define the loss function
  LossFunction ce_loss_fn = [](const torch::Tensor& predicted, const torch::Tensor& target) {
        return torch::nn::functional::cross_entropy(predicted, target);
    };
  // Train the model
  train_model(config, labels, features, ce_loss_fn, model);

  return 0;
}

int learnable_w(ConfigProperties& config){

  torch::Tensor dvh_coo_list = tensor_from_file<float>(config.data_properties.dvh_path);
  torch::Tensor dvh = coo_tensor_to_sparse(dvh_coo_list);
  std::cout << "DVH dimensions: " << dvh.sizes() << std::endl;

  torch::Tensor invde_ht_dvh_coo_list = tensor_from_file<float>(config.data_properties.invde_ht_dvh_path);
  torch::Tensor invde_ht_dvh = coo_tensor_to_sparse(invde_ht_dvh_coo_list);
  std::cout << "INVDE_HT_DVH dimensions: " << invde_ht_dvh.sizes() << std::endl;

  torch::Tensor labels = tensor_from_file<float>(config.data_properties.labels_path);
  labels = labels.index({at::indexing::Slice(), at::indexing::Slice(-1)}).squeeze().to(torch::kLong);
  std::cout << "labels shape: " << labels.sizes() << std::endl;
   
  torch::Tensor features = tensor_from_file<float>(config.data_properties.features_path);
  std::cout << "features shape: " << features.sizes() << std::endl;
  int f_cols = features.size(1);

  auto model = new ModelW(f_cols, config.model_properties.hidden_dims, config.model_properties.classes, config.model_properties.dropout_rate, &dvh, &invde_ht_dvh, config.model_properties.with_bias);

  // Define the loss function
  LossFunction ce_loss_fn = [](const torch::Tensor& predicted, const torch::Tensor& target) {
        return torch::nn::functional::cross_entropy(predicted, target);
    };

  // Train the model
  train_model(config, labels, features, ce_loss_fn, model);

  return 0;
}

int dist_learning(ConfigProperties& config) {

  torch::Tensor labels = tensor_from_file<float>(config.data_properties.labels_path);
  labels = labels.index({at::indexing::Slice(), at::indexing::Slice(-1)}).squeeze().to(torch::kLong);
  std::cout << "labels shape: " << labels.sizes() << std::endl;
   
  torch::Tensor features = tensor_from_file<float>(config.data_properties.features_path);
  std::cout << "features shape: " << features.sizes() << std::endl;
  int f_cols = features.size(1);

  auto model = new DistModel(config, f_cols);

  // Define the loss function
  LossFunction ce_loss_fn = [](const torch::Tensor& predicted, const torch::Tensor& target) {
        return torch::nn::functional::cross_entropy(predicted, target);
    };

  // Train the model
  // train_model(config, labels, features, ce_loss_fn, model);

  return 0;
}

int main(int argc, char** argv){

  // read command line arguments
  int opt;
  std::string config_path;
  std::string tmp_dir = "";
  while((opt = getopt(argc, argv, "c:t:")) != -1){
    switch(opt){
      case 'c':
        config_path = optarg;
        break;
      case 't':
        tmp_dir = optarg;
        break;
      default:
        std::cerr << "Usage: " << argv[0] << " -c <config_path>" << std::endl;
        exit(EXIT_FAILURE);
    }
  }

  // load config
  ConfigProperties config = ParseConfig(config_path);

  if (!tmp_dir.empty()) {
    config.data_properties.g_path = config.data_properties.g_path.replace(config.data_properties.g_path.find(".."), 2, tmp_dir);
    config.data_properties.labels_path = config.data_properties.labels_path.replace(config.data_properties.labels_path.find(".."), 2, tmp_dir);
    config.data_properties.features_path = config.data_properties.features_path.replace(config.data_properties.features_path.find(".."), 2, tmp_dir);
  }

  if (config.model_properties.learnable_w) {
    return learnable_w(config);
  } else {
    return model(config);
  }
}
