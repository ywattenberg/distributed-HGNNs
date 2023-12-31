#include <mpi.h>
#include <cblas.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <fast_matrix_market/fast_matrix_market.hpp>

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/SpParMat.h"
#include "CombBLAS/ParFriends.h"
#include "CombBLAS/FullyDistVec.h"
#include "CombBLAS/SpParMat.h"
#include "CombBLAS/DenseParMat.h"
#include "../DenseMatrix/DenseMatrix.h"

using namespace std;
using namespace combblas;

struct array_matrix {
    int64_t nrows = 0, ncols = 0;
    vector<double> vals;       // or int64_t, float, std::complex<double>, etc.
} mat;

const int ROWS_A = 4;  // Adjust this based on the actual size of your matrix
const int COLS_A = 5;  // Adjust this based on the actual size of your matrix
const int MATRIX_SIZE = ROWS_A * COLS_A;
const int ROWS_B = COLS_A;
const int COLS_B = 4;  // Adjust this based on the actual size of your matrix

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    {
    // create dense matrix
    const std::string filename = "../data/m_g_ms_gs/features.mtx";
	shared_ptr<CommGrid> fullWorld;
    fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );

    // SpParMat < int64_t, double, SpDCCols<int64_t, double> > A(fullWorld);
    // A.ParallelReadMM(filename, false, maximum<double>());
    // MPI_Barrier(MPI_COMM_WORLD);

    DenseMatrix<double> A(0, 0, fullWorld);
    A.ParallelReadDMM(filename, false);
    MPI_Barrier(MPI_COMM_WORLD);
    // print local matrix A
    // if (rank == 0) {
    //     A.printLocalMatrix();
    // }

    DenseMatrix<double> B(0, 0, fullWorld);
    B.ParallelReadDMM(filename, false);
    MPI_Barrier(MPI_COMM_WORLD);
    
    std::vector<double>* C = VPDGEMM(A.getValues(), B.getValues(), A.getLocalRows(), A.getLocalCols(), B.getLocalRows(), B.getLocalCols());

    // print first 100 elements of local matrix C
    if (rank == 0) {
        for (int i = 0; i < 10; i++) {
            std::cout << "Rank " << rank << " " << C->at(i) << " " << std::endl;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    }
    MPI_Finalize();
    return 0;
}