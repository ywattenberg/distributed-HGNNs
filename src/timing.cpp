#include <chrono>
#include <iostream>

#include "utils/DenseMatrix.h"

using namespace combblas;

int read_features() {
    const std::string filename = "../data/m_g_ms_gs/features.mtx";
    shared_ptr<CommGrid> fullWorld;
    fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );

    DenseMatrix<double> A(0, 0, fullWorld);
    A.ParallelReadDMM(filename, false);
    MPI_Barrier(MPI_COMM_WORLD);

    return 0;
}

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    {
      using std::chrono::high_resolution_clock;
      using std::chrono::duration_cast;
      using std::chrono::duration;
      using std::chrono::milliseconds;

      auto t1 = high_resolution_clock::now();
      if (rank == 0) {
        t1 = high_resolution_clock::now();
      }
      read_features();

      MPI_Barrier(MPI_COMM_WORLD);

      if (rank == 0) {
        auto t2 = high_resolution_clock::now();

        /* Getting number of milliseconds as an integer. */
        auto ms_int = duration_cast<milliseconds>(t2 - t1);

        /* Getting number of milliseconds as a double. */
        duration<double, std::milli> ms_double = t2 - t1;

        std::cout << ms_int.count() << "ms\n";
        std::cout << ms_double.count() << "ms\n";

        // append time to csv file
        std::ofstream outfile;
        outfile.open("../data/timing/read_feature.csv", std::ios_base::app);
        outfile << size << ", " << ms_int.count() << "\n";
        outfile.close();

      }
    }
    MPI_Finalize();
    return 0;
}