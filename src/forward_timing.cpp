#include <chrono>
#include <iostream>

#include <torch/torch.h>

#include "utils/DenseMatrix.h"
#include "model/model.h"
#include "utils/fileParse.h"

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

int forward_base_model(int run_id, int iterations, int nprocs, int scale, std::string tmp_dir, std::vector<int> hidden_dims, int classes, float dropout_rate, bool with_bias) {
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
    auto model = new Model(f_cols, hidden_dims, classes, dropout_rate, &left_side, with_bias);
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
      outfile << run_id << "," << model_name << "," << scale << "," << i << "," << ms_int.count() << "," << nprocs << "," << hidden_dims << "," << classes << "," << dropout_rate << "," << with_bias << "\n";
    }

    return 0;
}

int forward_learnable_w(int run_id, int iterations, int nprocs, int scale, std::string tmp_dir, std::vector<int> hidden_dims, int classes, float dropout_rate, bool with_bias) {   

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
    auto model = new ModelW(f_cols, hidden_dims, classes, dropout_rate, &dvh, &invde_ht_dvh, with_bias);
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
      outfile << run_id << "," << model_name << "," << scale << "," << i << "," << ms_int.count() << "," << nprocs << "," << hidden_dims << "," << classes << "," << dropout_rate << "," << with_bias << "\n";
    }
}

int main(int argc, char* argv[]) {

    int opt;
    std::string tmp_dir = "";
    int run_id = -1;
    int nprocs = -1;
    while((opt = getopt(argc, argv, "d:i:p:")) != -1){
      switch(opt){
        case 'd':
          tmp_dir = optarg;
          break;
        case 'i':
          run_id = atoi(optarg);
          break;
        case 'p':
          nprocs = atoi(optarg);
          break;
        default:
          std::cerr << "Usage: " << argv[0] << " -d <tmp_dir>" << std::endl;
          exit(EXIT_FAILURE);
      }
    }

    // MPI_Init(&argc, &argv);

    // int size, rank;
    // MPI_Comm_size(MPI_COMM_WORLD, &size);
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
    {
      int iterations = 10;
      const std::string data_dir = tmp_dir;
      std::vector<int> hidden_dims = {128};
      int classes = 40;
      float dropout_rate = 0.5;
      bool with_bias = true;

      for (int scale = 1; scale <= 8; scale *= 2) {
        forward_base_model(run_id, iterations, nprocs, scale, tmp_dir, hidden_dims, classes, dropout_rate, with_bias);
        forward_learnable_w(run_id, iterations, nprocs, scale, tmp_dir, hidden_dims, classes, dropout_rate, with_bias);
      }
    }

    // MPI_Finalize();
    return 0;
}