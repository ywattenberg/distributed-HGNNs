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
  // Create Model
  DistModel model(config, 10);
  MPI_Finalize();

}