#include <iostream>
#include <fstream>
#include <filesystem>
#include <unistd.h>
#include <sstream>
#include <filesystem>
namespace fs = std::filesystem;

#include <yaml-cpp/yaml.h>
#include <torch/torch.h>

#include "trainer/trainer.h"
#include "DenseMatrix/DenseMatrix.h"
#include "utils/configParse.h"
#include "model/dist-model.h"
#include "model/dist-model-w.h"
#include "model/model.h"
#include "utils/LossFn.h"
#include "utils/fileParse.h"
#include "trainer/dist-trainer.h"

std::vector<int> readCSV(const std::string& filename) {
    std::vector<int> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        // Handle error
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;

        while (std::getline(lineStream, cell, ',')) {
          try {
            int cell_to_int = std::stoi(cell);
            data.push_back(cell_to_int);
          } catch (const std::invalid_argument& e) {
              std::cerr << "Invalid argument: " << e.what() << std::endl;
          } catch (const std::out_of_range& e) {
              std::cerr << "Out of range: " << e.what() << std::endl;
          }
            
        }

    }

    file.close();
    return data;
}

inline torch::Tensor coo_tensor_to_sparse(torch::Tensor& coo_tensor){
  torch::Tensor index = coo_tensor.index({at::indexing::Slice(), at::indexing::Slice(0,2)}).transpose(0,1);
  index = index.to(torch::kLong);
  torch::Tensor values = coo_tensor.index({at::indexing::Slice(), at::indexing::Slice(2,3)}).squeeze();
  // This way of calling sparse_coo_tensor assumes 
  // that we have at least one non-zero value in each row/column 
  return torch::sparse_coo_tensor(index, values);
}

using LossFunction = at::Tensor(*)(const at::Tensor&, const at::Tensor&); //Supertype for loss functions

int model(ConfigProperties& config, bool timing, std::string run_id){
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
  
  std::string timing_file = "../data/timing/main_training.csv";

  if (timing) {
    std::ostringstream oss;
    oss << "../data/timing/main_training_info.csv";
    std::string info_file = oss.str();
    std::ofstream outfile;
    outfile.open(info_file, std::ios_base::app);
    outfile << run_id << "," << config.model_properties.learnable_w << "," << config.model_properties.hidden_dims << "," << config.model_properties.with_bias << "," << config.model_properties.dropout_rate << "," << config.trainer_properties.epochs << "," << config.data_properties.dataset << "\n";
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

  // if (timing) {
  //   std::ofstream outfile;
  //   outfile.open("../data/timing/" + std::to_string(run_id) + "/config.yaml", std::ios_base::app);
  //   outfile << "\ntraining_time: " << ms_int.count();
  // }

  return 0;
}

int learnable_w(ConfigProperties& config, bool timing, std::string run_id){

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
  
  std::string timing_file = "../data/timing/main_training.csv";

  if (timing) {
    std::ostringstream oss;
    oss << "../data/timing/main_training_info.csv";
    std::string info_file = oss.str();
    std::ofstream outfile;
    outfile.open(info_file, std::ios_base::app);
    outfile << run_id << "," << config.model_properties.learnable_w << "," << config.model_properties.hidden_dims << "," << config.model_properties.with_bias << "," << config.model_properties.dropout_rate << "," << config.trainer_properties.epochs << "," << config.data_properties.dataset << "\n";
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

  // if (timing) {
  //   std::ofstream outfile;
  //   outfile.open("../data/timing/" + std::to_string(run_id) + "/config.yaml", std::ios_base::app);
  //   outfile << "\ntraining_time: " << ms_int.count();
  // }

  return 0;
}

int dist_model(ConfigProperties& config, bool timing, std::string run_id) {

  int nprocs, myrank;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

  shared_ptr<CommGrid> fullWorld;
	fullWorld.reset(new CommGrid(MPI_COMM_WORLD, std::sqrt(nprocs), std::sqrt(nprocs)));

  std::string timing_file = "../data/timing/main_training.csv";

  // Create Model
  DistModel model(config, 6144, fullWorld, 24622);

  cout << myrank << ": model initialized" << endl;

  MPI_Barrier(MPI_COMM_WORLD);

  vector<int> labels = readCSV(config.data_properties.labels_path);
  labels.erase(labels.begin()); // delete first element (size)


  DenseMatrix<double> input(0, 0, fullWorld);
  input.ParallelReadDMM(config.data_properties.features_path, false);
  
  int totalRows = input.getnrow();
  int totalCols = input.getncol();

  if (myrank == 0){
    cout << "number of samples: " << totalRows << endl;
    cout << "number of features: " << totalCols << endl;
  }

  int test_idx = config.data_properties.test_idx;

  if (timing && myrank == 0) {
    std::ostringstream oss;
    oss << "../data/timing/main_training_info.csv";
    std::string info_file = oss.str();
    std::ofstream outfile;
    outfile.open(info_file, std::ios_base::app);
    outfile << run_id << "," << config.model_properties.learnable_w << "," << config.model_properties.hidden_dims << "," << config.model_properties.with_bias << "," << config.model_properties.dropout_rate << "," << config.trainer_properties.epochs << "," << config.data_properties.dataset << "\n";
    outfile.close();
  }

  train_dist_model(config, labels, input, &model, run_id, timing, timing_file);

  MPI_Finalize();

  return 0;
}

int main(int argc, char** argv){

  // read command line arguments
  int opt;
  std::string config_path;
  std::string tmp_dir = "";
  bool timing = false;
  std::string run_id = "-1_-1";
  while((opt = getopt(argc, argv, "c:d:i:t:")) != -1){
    switch(opt){
      case 'c':
        config_path = optarg;
        break;
      case 'd':
        tmp_dir = optarg;
        break;
      case 'i':
        run_id = optarg;
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

  std::cout << "Run id: " << run_id << std::endl;

  std::cout << "Config path: " << config_path << std::endl;
  // load config
  ConfigProperties config = ParseConfig(config_path);
  std::cout << "Config loaded" << std::endl;

  std::cout << "Tmp dir: " << tmp_dir << std::endl;
  if (!tmp_dir.empty()) {
    config.data_properties.g_path = config.data_properties.g_path.replace(config.data_properties.g_path.find(".."), 2, tmp_dir);
    config.data_properties.labels_path = config.data_properties.labels_path.replace(config.data_properties.labels_path.find(".."), 2, tmp_dir);
    config.data_properties.features_path = config.data_properties.features_path.replace(config.data_properties.features_path.find(".."), 2, tmp_dir);
    config.data_properties.dvh_path = config.data_properties.dvh_path.replace(config.data_properties.dvh_path.find(".."), 2, tmp_dir);
    config.data_properties.invde_ht_dvh_path = config.data_properties.invde_ht_dvh_path.replace(config.data_properties.invde_ht_dvh_path.find(".."), 2, tmp_dir);
  }

  if (config.model_properties.distributed) {
    return dist_model(config, timing, run_id);
  } else {
    if (config.model_properties.learnable_w) {
      return learnable_w(config, timing, run_id);
    } else {
      return model(config, timing, run_id);
    }
  }
}
