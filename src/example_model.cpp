#include <mpi.h>
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
#include "utils/LossFn.h"


typedef combblas::PlusTimesSRing<double, double> PTFF;

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
  

  shared_ptr<CommGrid> fullWorld;
	fullWorld.reset(new CommGrid(MPI_COMM_WORLD, std::sqrt(nprocs), std::sqrt(nprocs)));

  // Create Model
  DistModel model(config, 6144, fullWorld, 24622);

  cout << myrank << ": model initialized" << endl;

  MPI_Barrier(MPI_COMM_WORLD);

  DenseMatrix<double> input(0, 0, fullWorld);
  input.ParallelReadDMM(config.data_properties.features_path, false);

  int totalRows = input.getnrow();
  int totalCols = input.getncol();

  if (myrank == 0){
    cout << "number of features: " << totalRows << endl;
    cout << "number of samples: " << totalCols << endl;
  }

  std::vector<int> labels(totalCols, 0);

  MPI_Barrier(MPI_COMM_WORLD);

  DenseMatrix<double> res = model.forward(&input);
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "loss: " << CrossEntropyLoss<PTFF, double>(res, &labels) << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);
  model.backward(res, &labels, 0.1);
  MPI_Finalize();

}