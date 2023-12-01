#include <mpi.h>
#include <cblas.h>
#include <iostream>
#include <vector>

using namespace std;

double* PDGEMM(const double A[], const double B[], int ROWS_A, int COLS_B, int COLS_A) {

    double C[ROWS_A * COLS_B];

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		const int MATRIX_SIZE = ROWS_A * COLS_A;

		// assert that matrices are compatible
		// assert(COLS_A == ROWS_B);

    int rowsPerProcess = ROWS_A / size;
    int remainingRows = ROWS_A % size;

    int myRows = (rank < remainingRows) ? (rowsPerProcess + 1) : rowsPerProcess;
    int myOffset = (rank < remainingRows) ? rank * ((rowsPerProcess + 1) * COLS_A) : (remainingRows * (rowsPerProcess + 1) * COLS_A) + ((rank - remainingRows) * rowsPerProcess * COLS_A);

		// print all info for this thread in one line
    std::cout << "rank: " << rank << ", myRows: " << myRows << ", myOffset: " << myOffset << std::endl;

		double* localA = new double[myRows * COLS_A];
    double* localC = new double[myRows * COLS_B];

		MPI_Scatter(A, myRows * COLS_A, MPI_DOUBLE, localA, myRows * COLS_A, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,myRows,COLS_B,COLS_A,1,localA, 2, B, COLS_A,0,localC,2);

		// print the local portion of A
		for (int i = 0; i < myRows; i++) {
				for (int j = 0; j < COLS_A; j++) {
						std::cout << localA[i * COLS_A + j] << " ";
				}
				std::cout << std::endl;
		}

		// Gather the results back to the root process
		MPI_Gather(localC, myRows * COLS_B, MPI_DOUBLE, C, myRows * COLS_B, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// print the final result
		if (rank == 0) {
				for (int i = 0; i < ROWS_A; i++) {
						for (int j = 0; j < COLS_B; j++) {
								std::cout << C[i * COLS_B + j] << " ";
						}
						std::cout << std::endl;
				}
		}

		// check if result is correct
		if (rank == 0) {
				double* correctC = new double[ROWS_A * COLS_B];
				cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,ROWS_A,COLS_B,COLS_A,1,A, 2, B, COLS_A,0,correctC,2);
				for (int i = 0; i < ROWS_A; i++) {
						for (int j = 0; j < COLS_B; j++) {
								if (C[i * COLS_B + j] != correctC[i * COLS_B + j]) {
										std::cout << "wrong result" << std::endl;
										break;
								}
						}
				}
        std::cout << "correct result" << std::endl;
    }

    delete[] localC;
    delete[] localA;

		return C;
}