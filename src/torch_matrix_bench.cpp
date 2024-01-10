#include <mpi.h>
#include <cblas.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>

#include <torch/torch.h>

#include "utils/fileParse.h"
#include "DenseMatrix/DenseMatrix.h"
#include "DenseMatrix/DenseMatrixAlgorithms.h"


using namespace std;


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


int main(int argc, char* argv[]) {
    int opt;
    std::string tmp_dir;
    int run_id = -1;
    int iterations = 10;
    while((opt = getopt(argc, argv, "t:i:")) != -1){
      switch(opt){
        case 't':
          tmp_dir = optarg;
          break;
        case 'i':
          run_id = atoi(optarg);
          break;
        default:
          std::cerr << "Invalid command line argument" << std::endl;
          exit(1);
      }
    }

    {

      for (int scale = 1; scale <= 4; scale*=2) {

        std::string features_filename = tmp_dir + "/data/random/scale_" + std::to_string(scale) + "/features.csv";
        torch::Tensor features = tensor_from_file<float>(features_filename);
        std::cout << "features shape: " << features.sizes() << std::endl;

        torch::Tensor features2 = tensor_from_file<float>(features_filename);
        features2 = features2.t();

        using std::chrono::high_resolution_clock;
        using std::chrono::duration_cast;
        using std::chrono::duration;
        using std::chrono::milliseconds;

        for (int i = 0; i < iterations; i++) {
          auto t1 = high_resolution_clock::now();

          auto tmp = features.mm(features2);

          auto t2 = high_resolution_clock::now();
          /* Getting number of milliseconds as an integer. */
          auto ms_int = duration_cast<milliseconds>(t2 - t1);
          /* Getting number of milliseconds as a double. */
          duration<double, std::milli> ms_double = t2 - t1;

          std::cout <<"Forward took " << ms_int.count() << "ms\n";

          std::ofstream outfile;
          outfile.open("../data/timing/mm_bench.csv", std::ios_base::app);
          outfile << run_id << "," << scale << "," << i << "," << ms_int.count() << "\n";
        }
      }
    }
}