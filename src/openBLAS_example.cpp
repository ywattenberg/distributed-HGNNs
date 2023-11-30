// #include <mpi.h>
// #include <cblas.h>
// #include <stdio.h>

// int main()
// {

//   // Initialize the MPI environment
//   MPI_Init(NULL, NULL);

//   // Get the number of processes
//   int world_size;
//   MPI_Comm_size(MPI_COMM_WORLD, &world_size);

//   // Get the rank of the process
//   int world_rank;
//   MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

//   if (world_rank == 0)
//   {
//     printf("Hello world from processor %d of %d\n", world_rank, world_size);

//   }

//   int len_a = 4;
//   int i=0;
//   double A[4] = {7.0,5.0,6.0,3.0};
//   double B[4] = {2.0,1.0,5.0,1.0};
//   double C[4] = {1.,1.,1.,1.};

//   int 

//   cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,2,2,2,1,A, 2, B, 2,0,C,2);

//   for(i=0; i<4; i++)
//     printf("%lf ", C[i]);
//   printf("\n");

//   MPI_Finalize();

//   return 0;
// }

#include <mpi.h>
#include <cblas.h>
#include <iostream>
#include <vector>

const int ROWS_A = 4;  // Adjust this based on the actual size of your matrix
const int COLS_A = 2;  // Adjust this based on the actual size of your matrix
const int MATRIX_SIZE = ROWS_A * COLS_A;
const int ROWS_B = COLS_A;
const int COLS_B = 2;  // Adjust this based on the actual size of your matrix

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double B[ROWS_B * COLS_B] = {2.0,1.0,5.0,1.0};
    double C[ROWS_A * COLS_B] = {1.,1.,1.,1.,1.,1.,1.,1.};
    double* A = new double[MATRIX_SIZE];

    if (rank == 0) {
        // Assuming A is your matrix, initialize it here
        // A = {7.0,5.0,6.0,3.0};
        A[0] = 7.0;
        A[1] = 5.0;
        A[2] = 6.0;
        A[3] = 3.0;
        A[4] = 7.0;
        A[5] = 5.0;
        A[6] = 6.0;
        A[7] = 3.0;

    }
    int rowsPerProcess = ROWS_A / size;
    int remainingRows = ROWS_A % size;

    int myRows = (rank < remainingRows) ? (rowsPerProcess + 1) : rowsPerProcess;
    int myOffset = (rank < remainingRows) ? rank * ((rowsPerProcess + 1) * COLS_A) : (remainingRows * (rowsPerProcess + 1) * COLS_A) + ((rank - remainingRows) * rowsPerProcess * COLS_A);

    // print all info for this thread in one line
    std::cout << "rank: " << rank << ", myRows: " << myRows << ", myOffset: " << myOffset << std::endl;

    // Allocate memory for the local portion of A
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

    // // print the local portion of C
    // for (int i = 0; i < myRows; i++) {
    //     for (int j = 0; j < COLS_B; j++) {
    //         std::cout << localC[i * COLS_B + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

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

    MPI_Finalize();
    return 0;
}
