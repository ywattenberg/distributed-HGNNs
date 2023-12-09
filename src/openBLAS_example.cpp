#include <mpi.h>
#include <cblas.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <fast_matrix_market/fast_matrix_market.hpp>

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/SpParMat3D.h"
#include "CombBLAS/ParFriends.h"
#include "CombBLAS/FullyDistVec.h"
#include "CombBLAS/SpParMat.h"
#include "CombBLAS/DenseParMat.h"
#include "utils/parDenseGEMM.h"
#include "utils/DenseMatrix.h"

using namespace std;
using namespace combblas;

struct array_matrix {
    int64_t nrows = 0, ncols = 0;
    vector<double> vals;       // or int64_t, float, std::complex<double>, etc.
} mat;

vector<double>* vals;

const int ROWS_A = 4;  // Adjust this based on the actual size of your matrix
const int COLS_A = 4;  // Adjust this based on the actual size of your matrix
const int MATRIX_SIZE = ROWS_A * COLS_A;
const int ROWS_B = COLS_A;
const int COLS_B = 4;  // Adjust this based on the actual size of your matrix

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // create dense matrix
    const std::string filename = "../data/m_g_ms_gs/dense-test.mtx";
	shared_ptr<CommGrid> fullWorld;
	fullWorld.reset(new CommGrid(MPI_COMM_WORLD, std::sqrt(size), std::sqrt(size)));
    DenseMatrix<double> A(ROWS_A, COLS_B, vals, fullWorld);
    A.ParallelReadDMM(filename, false);
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "rank " << rank << " read dense matrix" << std::endl;
    // print local matrix A
    std::vector<double>* vals = A.getValues();
    for (int i = 0; i < vals->size(); i++) {
        std::cout << "rank " << rank << " local matrix A[" << i << "] = " << vals->at(i) << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}