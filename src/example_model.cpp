#include <mpi.h>
#include <omp.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/SpParMat3D.h"
#include "CombBLAS/ParFriends.h"
#include "CombBLAS/FullyDistVec.h"
#include "CombBLAS/SpParMat.h"
#include "utils/DenseMatrix.h"
#include "utils/configParse.h"
#include "model/dist-model.h"


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



int main(int argc, char* argv[]){
   int nprocs, myrank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  
  if(argc < 2){
    if(myrank == 0)
    {
      cout << "Usage: ./<Binary> <ConfigFile>" << endl;
    }
    MPI_Finalize();
    return -1;
  }
  ConfigProperties config = ParseConfig(argv[1]);
  if(myrank == 0)
  {
    cout << "Read config " << argv[1] << endl;
  }
  
  int num_threads = 1;

  shared_ptr<CommGrid> fullWorld;
	fullWorld.reset(new CommGrid(MPI_COMM_WORLD, std::sqrt(nprocs), std::sqrt(nprocs)));

  // Create Model
  DistModel model(config, 6144, fullWorld, 24622);

  cout << myrank << ": model initialized" << endl;

  MPI_Barrier(MPI_COMM_WORLD);

  vector<int> labels = readCSV(config.data_properties.labels_path);
  labels.erase(labels.begin()); // delete first element (size)

  if (myrank == 0){
    for (int i = 0; i < 10; i++){
      cout << "label: " << labels[i] << endl;
    }
  }

  // cout << "label: " << labels.at(myrank * 1000) << endl;

  MPI_Barrier(MPI_COMM_WORLD);

  DenseMatrix<double> input(0, 0, fullWorld);
  input.ParallelReadDMM(config.data_properties.features_path, false);

  int totalRows = input.getnrow();
  int totalCols = input.getncol();

  if (myrank == 0){
    cout << "number of features: " << totalRows << endl;
    cout << "number of samples: " << totalCols << endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  auto outForward = model.forward(&input);

  cout << "forward done" << endl;

  MPI_Barrier(MPI_COMM_WORLD);

  // model.backward(input, &labels, 0.01);

  

  MPI_Finalize();

}