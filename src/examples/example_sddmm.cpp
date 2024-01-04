#include <string>
#include "SpmatLocal.hpp"


int main(int argc, char* argv[]){

  int nprocs, myrank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

  std::string fname(argv[1]);
  SpmatLocal S;
  S.loadTuples(true, -1, -1, fname);

  std::cout << "Hello " << std::endl;

  MPI_Finalize();

}