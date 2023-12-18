#include <iostream>
#include <fstream>
#include <filesystem>
#include <unistd.h>
#include <sstream>
#include <filesystem>
namespace fs = std::filesystem;

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

int model(ConfigProperties& config, bool timing, int run_id, int cpus){
  std::cout << "Running model" << std::endl;
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
  
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;

  auto t1 = high_resolution_clock::now();
  
  std::string timing_file = "";

  if (timing) {
    std::ostringstream oss;
    oss << "../data/timing/" << run_id << "/base_model.csv";
    timing_file = oss.str();
    std::ofstream outfile;
    outfile.open(timing_file, std::ios_base::app);
    outfile << "run_id,epoch,epoch_time,train_loss,test_loss,test_acc,test_f1\n";
    outfile.close();
  }

  // Train the model
  train_model(config, labels, features, ce_loss_fn, model, run_id, timing, timing_file);

  auto t2 = high_resolution_clock::now();
  /* Getting number of milliseconds as an integer. */
  auto ms_int = duration_cast<milliseconds>(t2 - t1);
  /* Getting number of milliseconds as a double. */
  duration<double, std::milli> ms_double = t2 - t1;

  std::cout <<"Training took " << ms_int.count() << "ms\n";

  if (timing) {
    std::ofstream outfile;
    outfile.open("../data/timing/" + std::to_string(run_id) + "/config.yaml", std::ios_base::app);
    outfile << "\ntraining_time: " << ms_int.count();
  }

  return 0;
}

int learnable_w(ConfigProperties& config, bool timing, int run_id, int cpus){

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
  
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;

  auto t1 = high_resolution_clock::now();
  
  std::string timing_file = "";

  if (timing) {
    std::ostringstream oss;
    oss << "../data/timing/" << run_id << "/base_model.csv";
    timing_file = oss.str();
    std::ofstream outfile;
    outfile.open(timing_file, std::ios_base::app);
    outfile << "run_id,epoch,epoch_time,train_loss,test_loss,test_acc,test_f1\n";
    outfile.close();
  }
  // Train the model
  train_model(config, labels, features, ce_loss_fn, model, run_id, timing, timing_file);

  auto t2 = high_resolution_clock::now();
  /* Getting number of milliseconds as an integer. */
  auto ms_int = duration_cast<milliseconds>(t2 - t1);
  /* Getting number of milliseconds as a double. */
  duration<double, std::milli> ms_double = t2 - t1;

  std::cout <<"Training took " << ms_int.count() << "ms\n";

  if (timing) {
    std::ofstream outfile;
    outfile.open("../data/timing/" + std::to_string(run_id) + "/config.yaml", std::ios_base::app);
    outfile << "\ntraining_time: " << ms_int.count();
  }

  return 0;
}

int dist_learning(ConfigProperties& config) {

  torch::Tensor labels = tensor_from_file<float>(config.data_properties.labels_path);
  labels = labels.index({at::indexing::Slice(), at::indexing::Slice(-1)}).squeeze().to(torch::kLong);
  std::cout << "labels shape: " << labels.sizes() << std::endl;
   
  torch::Tensor features = tensor_from_file<float>(config.data_properties.features_path);
  std::cout << "features shape: " << features.sizes() << std::endl;
  int f_cols = features.size(1);

  // auto model = new DistModel(config, f_cols);

  // Define the loss function
  // LossFunction ce_loss_fn = [](const torch::Tensor& predicted, const torch::Tensor& target) {
  //       return torch::nn::functional::cross_entropy(predicted, target);
  //   };

  // Train the model
  // train_model(config, labels, features, ce_loss_fn, model);

  return 0;
}

int main(int argc, char** argv){

  // read command line arguments
  int opt;
  std::string config_path;
  std::string tmp_dir = "";
  bool timing = false;
  int run_id = -1;
  int cpus = 1;
  while((opt = getopt(argc, argv, "c:d:i:p:t:")) != -1){
    switch(opt){
      case 'c':
        config_path = optarg;
        break;
      case 'd':
        tmp_dir = optarg;
        break;
      case 'i':
        run_id = atoi(optarg);
        break;
      case 'p':
        // std::cout << "type of p " << optarg << std::endl;
        // std::cout << "Using " << atoi(optarg) << " cpus" << std::endl;
        cpus = atoi(optarg);
        break;
      case 't':
        if (atoi(optarg) == 1) {
          timing = true;
        } else {
          timing = false;
        }
        break;
      default:
        std::cerr << "Usage: " << argv[0] << " -c <config_path>" << std::endl;
        exit(EXIT_FAILURE);
    }
  }

  std::cout << "Using " << cpus << " cpus" << std::endl;
  std::cout << "Run id: " << run_id << std::endl;

  std::cout << "Config path: " << config_path << std::endl;
  // load config
  ConfigProperties config = ParseConfig(config_path);
  std::cout << "Config loaded" << std::endl;

  if (!tmp_dir.empty()) {
    config.data_properties.g_path = config.data_properties.g_path.replace(config.data_properties.g_path.find(".."), 2, tmp_dir);
    config.data_properties.labels_path = config.data_properties.labels_path.replace(config.data_properties.labels_path.find(".."), 2, tmp_dir);
    config.data_properties.features_path = config.data_properties.features_path.replace(config.data_properties.features_path.find(".."), 2, tmp_dir);
  }

  if(timing) {
    fs::create_directories("../data/timing/" + std::to_string(run_id));
    fs::copy_file(config_path, "../data/timing/" + std::to_string(run_id) + "/config.yaml");
    std::ofstream outfile;
    outfile.open("../data/timing/" + std::to_string(run_id) + "/config.yaml", std::ios_base::app);
    outfile << "\nrun_id: " << run_id;
    outfile << "\ncpus: " << cpus;
    outfile.close();
  }
  if (config.model_properties.learnable_w) {
    return learnable_w(config, timing, run_id, cpus);
  } else {
    return model(config, timing, run_id, cpus);
  }
}
