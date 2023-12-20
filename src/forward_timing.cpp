#include <chrono>
#include <iostream>

#include <torch/torch.h>

#include "utils/DenseMatrix.h"
#include "model/model.h"
#include "utils/fileParse.h"
#include "utils/configParse.h"
#include "model/dist-model.h"

using namespace combblas;

inline torch::Tensor coo_tensor_to_sparse(torch::Tensor& coo_tensor){
  torch::Tensor index = coo_tensor.index({at::indexing::Slice(), at::indexing::Slice(0,2)}).transpose(0,1);
  index = index.to(torch::kLong);
  torch::Tensor values = coo_tensor.index({at::indexing::Slice(), at::indexing::Slice(2,3)}).squeeze();
  // This way of calling sparse_coo_tensor assumes 
  // that we have at least one non-zero value in each row/column 
  return torch::sparse_coo_tensor(index, values);
}

using LossFunction = at::Tensor(*)(const at::Tensor&, const at::Tensor&); //Supertype for loss functions

int forward_base_model(int run_id, int iterations, ConfigProperties& config, int nprocs, int scale, std::string tmp_dir) {
  std::string data_folder = tmp_dir + "/data";
  std::string model_name = "base_model";
  std::cout << "Running base model" << std::endl;
  const std::string g_filename = data_folder + "/random/scale_" + std::to_string(scale) + "/G.csv";
  torch::Tensor coo_list = tensor_from_file<float>(g_filename);
  torch::Tensor left_side = coo_tensor_to_sparse(coo_list);
  std::cout << "G dimensions: " << left_side.sizes() << std::endl;

  const std::string labels_filename = data_folder + "/m_g_ms_gs/labels.csv";
  torch::Tensor labels = tensor_from_file<float>(labels_filename);
  labels = labels.index({at::indexing::Slice(), at::indexing::Slice(-1)}).squeeze().to(torch::kLong);
  std::cout << "labels shape: " << labels.sizes() << std::endl;
  
  const std::string features_filename = data_folder + "/random/scale_" + std::to_string(scale) + "/features.csv";
  std::cout << "features_filename: " << features_filename << std::endl;
  torch::Tensor features = tensor_from_file<float>(features_filename);
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

  for (int i = 0; i < iterations; i++) {
    auto t1 = high_resolution_clock::now();

    torch::Tensor predictions = model->forward(features);

    auto t2 = high_resolution_clock::now();
    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout <<"Forward took " << ms_int.count() << "ms\n";
    
    std::ofstream outfile;
    outfile.open("../data/timing/forward_timing.csv", std::ios_base::app);
    outfile << run_id << "," << model_name << "," << scale << "," << i << "," << ms_int.count() << "," << nprocs << "," << config.model_properties.hidden_dims << "," << config.model_properties.classes << "," << config.model_properties.dropout_rate << "," << config.model_properties.with_bias << "\n";
  }

  return 0;
}

int forward_learnable_w(int run_id, int iterations, ConfigProperties& config, int nprocs, int scale, std::string tmp_dir) {   

  std::string data_folder = tmp_dir + "/data";
  std::string model_name = "learnable_w";
  std::cout << "Running learnable_w model" << std::endl;

  const std::string dvh_filename = data_folder + "/random/scale_" + std::to_string(scale) + "/DVH.csv";
  torch::Tensor dvh_coo_list = tensor_from_file<float>(dvh_filename);
  torch::Tensor dvh = coo_tensor_to_sparse(dvh_coo_list);
  std::cout << "DVH dimensions: " << dvh.sizes() << std::endl;

  const std::string invde_ht_dvh_filename = data_folder + "/random/scale_" + std::to_string(scale) + "/invDE_HT_DVH.csv";
  torch::Tensor invde_ht_dvh_coo_list = tensor_from_file<float>(invde_ht_dvh_filename);
  torch::Tensor invde_ht_dvh = coo_tensor_to_sparse(invde_ht_dvh_coo_list);
  std::cout << "INVDE_HT_DVH dimensions: " << invde_ht_dvh.sizes() << std::endl;

  const std::string labels_filename = data_folder + "/m_g_ms_gs/labels.csv";
  torch::Tensor labels = tensor_from_file<float>(labels_filename);
  labels = labels.index({at::indexing::Slice(), at::indexing::Slice(-1)}).squeeze().to(torch::kLong);
  std::cout << "labels shape: " << labels.sizes() << std::endl;
  
  const std::string features_filename = data_folder + "/random/scale_" + std::to_string(scale) + "/features.csv";
  torch::Tensor features = tensor_from_file<float>(features_filename);
  std::cout << "features shape: " << features.sizes() << std::endl;
  int f_cols = features.size(1);
  // Build Model
  auto model = new ModelW(f_cols, config.model_properties.hidden_dims, config.model_properties.classes, config.model_properties.dropout_rate, &dvh, &invde_ht_dvh, config.model_properties.with_bias);
  // Define the loss function
  LossFunction ce_loss_fn = [](const torch::Tensor& predicted, const torch::Tensor& target) {
        return torch::nn::functional::cross_entropy(predicted, target);
    };

  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;

  for (int i = 0; i < iterations; i++) {
    auto t1 = high_resolution_clock::now();

    torch::Tensor predictions = model->forward(features);

    auto t2 = high_resolution_clock::now();
    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout <<"Forward took " << ms_int.count() << "ms\n";
    
    std::ofstream outfile;
    outfile.open("../data/timing/forward_timing.csv", std::ios_base::app);
    outfile << run_id << "," << model_name << "," << scale << "," << i << "," << ms_int.count() << "," << nprocs << "," << config.model_properties.hidden_dims << "," << config.model_properties.classes << "," << config.model_properties.dropout_rate << "," << config.model_properties.with_bias << "\n";
  }

  return 0;
}

int forward_dist_model(int run_id, int iterations, ConfigProperties& config, int nprocs, int scale, std::string tmp_dir) {

  std::string data_folder = tmp_dir + "/data";
  std::string model_name = "dist_model";
  config.data_properties.dvh_path = data_folder + "/random/scale_" + std::to_string(scale) + "/DVH.mtx";
  config.data_properties.invde_ht_dvh_path = data_folder + "/random/scale_" + std::to_string(scale) + "/invDE_HT_DVH.mtx";
  config.data_properties.labels_path = data_folder + "/m_g_ms_gs/labels.csv";
  config.data_properties.features_path = data_folder + "/random/scale_" + std::to_string(scale) + "/features.mtx";

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int num_threads = 1;

  shared_ptr<CommGrid> fullWorld;
  fullWorld.reset(new CommGrid(MPI_COMM_WORLD, std::sqrt(size), std::sqrt(size)));

  DenseMatrix<double> input(0, 0, fullWorld);
  input.ParallelReadDMM(config.data_properties.features_path, false);

  int totalRows = input.getnrow();
  int totalCols = input.getncol();

  if (rank == 0){
    cout << "number of features: " << totalRows << endl;
    cout << "number of samples: " << totalCols << endl;
  }
  // Create Model
  DistModel model(config, totalRows, fullWorld);

  cout << rank << ": model initialized" << endl;

  MPI_Barrier(MPI_COMM_WORLD);

  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;

  for (int i = 0; i < iterations; i++) {

    MPI_Barrier(MPI_COMM_WORLD);

    auto t1 = high_resolution_clock::now();

    DenseMatrix<double> res = model.forward(&input);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
      auto t2 = high_resolution_clock::now();
      /* Getting number of milliseconds as an integer. */
      auto ms_int = duration_cast<milliseconds>(t2 - t1);
      /* Getting number of milliseconds as a double. */
      duration<double, std::milli> ms_double = t2 - t1;

      std::cout <<"Forward took " << ms_int.count() << "ms\n";
      
      std::ofstream outfile;
      outfile.open("../data/timing/forward_timing.csv", std::ios_base::app);
      outfile << run_id << "," << model_name << "," << scale << "," << i << "," << ms_int.count() << "," << nprocs << "," << config.model_properties.hidden_dims << "," << config.model_properties.classes << "," << config.model_properties.dropout_rate << "," << config.model_properties.with_bias << "\n";
      outfile.close();
    }
  }
}

int main(int argc, char* argv[]) {

  int opt;
  std::string tmp_dir = "";
  std::string config_path = "";
  int run_id = -1;
  int nprocs = -1;
  bool distributed = false;
  while((opt = getopt(argc, argv, "c:t:i:p:d:")) != -1){
    switch(opt){
      case 'c':
        config_path = optarg;
        break;
      case 't':
        tmp_dir = optarg;
        break;
      case 'i':
        run_id = atoi(optarg);
        break;
      case 'p':
        nprocs = atoi(optarg);
        break;
      case 'd':
        if (atoi(optarg) == 0) {
          distributed = false;
        } else {
          distributed = true;
        }
        break;
      default:
        std::cerr << "Usage: " << argv[0] << " -d <tmp_dir>" << std::endl;
        exit(EXIT_FAILURE);
    }
  }
  
  if (distributed) {
    MPI_Init(&argc, &argv);
  }

  std::cout << "Config path: " << config_path << std::endl;
  // load config
  ConfigProperties config = ParseConfig(config_path);
  std::cout << "Config loaded" << std::endl;

  {
    int iterations = 10;
    const std::string data_dir = tmp_dir;
    std::vector<int> hidden_dims = {128};
    int classes = 40;
    float dropout_rate = 0.5;
    bool with_bias = true;

    for (int scale = 1; scale <= 8; scale *= 2) {
      if (distributed) {
        forward_dist_model(run_id, iterations, config, nprocs, scale, tmp_dir);
      } else {
        forward_base_model(run_id, iterations, config, nprocs, scale, tmp_dir);
        forward_learnable_w(run_id, iterations, config, nprocs, scale, tmp_dir);
      }
    }
  }

  if (distributed) {
    MPI_Finalize();
  }

  return 0;
}