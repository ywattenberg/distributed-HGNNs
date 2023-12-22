#include "../DenseMatrix/DenseMatrixAlgorithms.h"

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <omp.h>
#include <cblas.h>
// #include <mkl.h>

#include "DenseMatrix.h"

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/SpParMat.h"
#include "CombBLAS/ParFriends.h"
#include "CombBLAS/SpParHelper.h"
#include "CombBLAS/SpDCCols.h"

namespace combblas {

typedef SpDCCols <int64_t, double> DCCols;
typedef SpParMat <int64_t, double, DCCols> MPI_DCCols;
typedef SpParMat<int64_t, double, SpDCCols<int64_t, double>> SPMAT_DOUBLE;
typedef DenseMatrix<double> DENSE_DOUBLE;
typedef PlusTimesSRing<double, double> PTFF;


template DenseMatrix<double> DenseDenseMult<PTFF, double>(DenseMatrix<double> &A, DenseMatrix<double> &B);

template<typename SR, typename NT>
DenseMatrix<NT> DenseDenseMult(DenseMatrix<NT> &A, DenseMatrix<NT> &B)
{
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);


  int stages, dummy;

  std::shared_ptr<CommGrid> GridC = ProductGrid((A.getCommGrid()).get(), (B.getCommGrid()).get(), stages, dummy, dummy);		

  std::vector<NT> * bufferA = new std::vector<NT>();
  std::vector<int> essentialsA(3);
  std::vector<NT> * bufferB = new std::vector<NT>();
  std::vector<int> essentialsB(3);

  int rankAinRow = A.getCommGrid()->GetRankInProcRow();
  int rankBinCol = B.getCommGrid()->GetRankInProcCol();

  int localRowsA = A.getLocalRows();
  int localColsA = A.getLocalCols();
  int localRowsB = B.getLocalRows();
  int localColsB = B.getLocalCols();

  std::vector<NT> * localOut = new std::vector<NT>(localRowsA * localColsB, 0.0);

  for (int i = 0; i < stages; i++){
    int sendingRank = i;
    
    if (rankAinRow == sendingRank){
      bufferA = A.getValues();

      essentialsA[1] = localRowsA;
      essentialsA[2] = localColsA;
      essentialsA[0] = essentialsA[1] * essentialsA[2];
    }

    BCastMatrixDense(GridC->GetRowWorld(), bufferA, essentialsA, sendingRank);


    if (rankBinCol == sendingRank){
      bufferB = B.getValues();

      essentialsB[1] = localRowsB;
      essentialsB[2] = localColsB;
      essentialsB[0] = essentialsB[1] * essentialsB[2];
    }

    BCastMatrixDense(GridC->GetColWorld(), bufferB, essentialsB, sendingRank);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,essentialsA[1],essentialsB[2],essentialsA[2],1.0,bufferA->data(), essentialsA[2], bufferB->data(), essentialsB[2],1.0,localOut->data(),essentialsB[2]);


    // blockDenseDense<SR, NT>(essentialsA[1], essentialsA[2], essentialsB[1], essentialsB[2], bufferA, bufferB, localOut);
  
  }

  return DenseMatrix<NT>(localRowsA, localColsB, localOut, GridC);
}


template void blockDenseDenseTrans<PTFF, double>(size_t rowsA, size_t colsA, size_t rowsB, size_t colsB, std::vector<double>* dense_A, std::vector<double>* dense_B, std::vector<double>* outValues);
// computes locally A * B^T, with A and B given
template<typename SR, typename NT>
void blockDenseDenseTrans(size_t rowsA, size_t colsA, size_t rowsB, size_t colsB, std::vector<NT>* dense_A, std::vector<NT>* dense_B, std::vector<NT>* outValues)
{
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  if (colsA != colsB) {
    throw std::invalid_argument("DIMENSIONS DON'T MATCH");
  }

  if (rowsA * rowsB != outValues->size()) {
    throw std::invalid_argument("DIMENSIONS DON'T MATCH 2");
  }

  #pragma omp parallel for
    for (size_t i = 0; i < rowsA; i++){
      for (size_t j = 0; j < rowsB; j++){
        NT sum = 0.0;
        for (size_t k = 0; k < colsA; k++){
          sum = SR::add(sum, SR::multiply(dense_A->at(i * colsA + k), dense_B->at(j * colsB + k)));
        }
        outValues->at(i * rowsB + j) = SR::add(outValues->at(i * rowsB + j), sum);
      }
    }
}


template DenseMatrix<double> DenseDenseTransMult<PTFF, double>(DenseMatrix<double> &A, DenseMatrix<double> &B);
// computes A*B^T distributed
template<typename SR, typename NT>
DenseMatrix<NT> DenseDenseTransMult(DenseMatrix<NT> &A, DenseMatrix<NT> &B)
{
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);


  int stages, dummy;

  std::shared_ptr<CommGrid> GridC = ProductGrid((A.getCommGrid()).get(), (B.getCommGrid()).get(), stages, dummy, dummy);		

  std::vector<NT> * bufferA = new std::vector<NT>();
  std::vector<int> essentialsA(3);
  std::vector<NT> * bufferB = new std::vector<NT>();
  std::vector<int> essentialsB(3);

  int rankAinRow = A.getCommGrid()->GetRankInProcRow();
  int rankBinCol = B.getCommGrid()->GetRankInProcCol();
  int rankBinRow = B.getCommGrid()->GetRankInProcRow();


  int localRowsA = A.getLocalRows();
  int localColsA = A.getLocalCols();
  int localRowsB = B.getLocalRows();
  int localColsB = B.getLocalCols();

  std::vector<NT> * localOut = new std::vector<NT>();

  int diagNeighbour = B.getCommGrid()->GetComplementRank();

  int transposeRows, transposeCols;
  std::vector<NT> *transposeValues = new std::vector<NT>();

  if (rankBinRow != rankBinCol){
    MPI_Status status;
    MPI_Sendrecv(&localRowsB, 1, MPI_INT, diagNeighbour, 0, &transposeRows, 1, MPI_INT, diagNeighbour, 0, B.getCommGrid()->GetWorld(), &status);
    MPI_Sendrecv(&localColsB, 1, MPI_INT, diagNeighbour, 0, &transposeCols, 1, MPI_INT, diagNeighbour, 0, B.getCommGrid()->GetWorld(), &status);

    transposeValues->resize(transposeRows * transposeCols);
    localOut->resize(localRowsA * transposeRows, 0.0);


    MPI_Sendrecv(B.getValues()->data(), localRowsB * localColsB, MPIType<NT>(), diagNeighbour, 0, transposeValues->data(), transposeRows * transposeCols, MPIType<NT>(), diagNeighbour, 0, B.getCommGrid()->GetWorld(), &status);

  } else {
    localOut->resize(localRowsA * localRowsB, 0.0);

  }

  for (int i = 0; i < stages; i++){
    int sendingRank = i;
    
    if (rankAinRow == sendingRank){
      bufferA = A.getValues();

      essentialsA[1] = localRowsA;  
      essentialsA[2] = localColsA;
      essentialsA[0] = essentialsA[1] * essentialsA[2];
    }

    BCastMatrixDense(GridC->GetRowWorld(), bufferA, essentialsA, sendingRank);


    if (rankBinCol == sendingRank){
      if (rankBinRow != rankBinCol){
        bufferB = transposeValues;
        essentialsB[1] = transposeRows;
        essentialsB[2] = transposeCols;
        essentialsB[0] = essentialsB[1] * essentialsB[2];
      } else {
        bufferB = B.getValues();

        essentialsB[1] = localRowsB;
        essentialsB[2] = localColsB;
        essentialsB[0] = essentialsB[1] * essentialsB[2];
      }
    }
    BCastMatrixDense(GridC->GetColWorld(), bufferB, essentialsB, sendingRank);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,essentialsA[1],essentialsB[1],essentialsA[2],1.0,bufferA->data(), essentialsA[2], bufferB->data(), essentialsB[2],1.0,localOut->data(),essentialsB[1]);

    // blockDenseDenseTrans<SR, NT>(essentialsA[1], essentialsA[2], essentialsB[1], essentialsB[2], bufferA, bufferB, localOut);
  
  }

  delete transposeValues;

  if (rankBinRow != rankBinCol){
    return DenseMatrix<NT>(localRowsA, transposeRows, localOut, GridC);
  } else {
    return DenseMatrix<NT>(localRowsA, localRowsB, localOut, GridC);  
  }
}

template DenseMatrix<double> DenseTransDenseMult<PTFF, double>(DenseMatrix<double> &A, DenseMatrix<double> &B);
// computes A^T * B, given the dense matrix A and dense matrix B
template<typename SR, typename NT>
DenseMatrix<NT> DenseTransDenseMult(DenseMatrix<NT> &A, DenseMatrix<NT> &B)
{
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);


  int stages, dummy;

  std::shared_ptr<CommGrid> GridC = ProductGrid((A.getCommGrid()).get(), (B.getCommGrid()).get(), stages, dummy, dummy);		

  std::vector<NT> * bufferA = new std::vector<NT>();
  std::vector<int> essentialsA(3);
  std::vector<NT> * bufferB = new std::vector<NT>();
  std::vector<int> essentialsB(3);

  int rankAinRow = A.getCommGrid()->GetRankInProcRow();
  int rankBinCol = B.getCommGrid()->GetRankInProcCol();
  int rankAinCol = A.getCommGrid()->GetRankInProcCol();


  int localRowsA = A.getLocalRows();
  int localColsA = A.getLocalCols();
  int localRowsB = B.getLocalRows();
  int localColsB = B.getLocalCols();

  

  int diagNeighbour = A.getCommGrid()->GetComplementRank();

  int transposeRows, transposeCols;
  std::vector<NT> *transposeValues = new std::vector<NT>();

  if (rankAinRow != rankAinCol){
    MPI_Status status;
    MPI_Sendrecv(&localRowsA, 1, MPI_INT, diagNeighbour, 0, &transposeRows, 1, MPI_INT, diagNeighbour, 0, A.getCommGrid()->GetWorld(), &status);
    MPI_Sendrecv(&localColsA, 1, MPI_INT, diagNeighbour, 0, &transposeCols, 1, MPI_INT, diagNeighbour, 0, A.getCommGrid()->GetWorld(), &status);

    transposeValues->resize(transposeRows * transposeCols);
    
    MPI_Sendrecv(A.getValues()->data(), localRowsA * localColsA, MPIType<NT>(), diagNeighbour, 0, transposeValues->data(), transposeRows * transposeCols, MPIType<NT>(), diagNeighbour, 0, A.getCommGrid()->GetWorld(), &status);
    
    localRowsA = transposeRows;
    localColsA = transposeCols;
  }

  

  std::vector<NT> * localOut = new std::vector<NT>(localColsA * localColsB, 0.0);


  for (int i = 0; i < stages; i++){
    int sendingRank = i;
    
    if (rankAinRow == sendingRank){
      if (rankAinRow != rankAinCol){
        bufferA = transposeValues;
      } else {
        bufferA = A.getValues();
      }

      essentialsA[1] = localRowsA;  
      essentialsA[2] = localColsA;
      essentialsA[0] = essentialsA[1] * essentialsA[2];
    }

    BCastMatrixDense(GridC->GetRowWorld(), bufferA, essentialsA, sendingRank);


    if (rankBinCol == sendingRank){
        bufferB = B.getValues();

        essentialsB[1] = localRowsB;
        essentialsB[2] = localColsB;
        essentialsB[0] = essentialsB[1] * essentialsB[2];
      
    }
    BCastMatrixDense(GridC->GetColWorld(), bufferB, essentialsB, sendingRank);

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,essentialsA[2],essentialsB[2],essentialsA[1],1.0,bufferA->data(), essentialsA[2], bufferB->data(), essentialsB[2], 1.0,localOut->data(),essentialsB[2]);
    // blockDenseDenseTrans<SR, NT>(essentialsA[1], essentialsA[2], essentialsB[1], essentialsB[2], bufferA, bufferB, localOut);


  }

  delete transposeValues;

  return DenseMatrix<NT>(localColsA, localColsB, localOut, GridC);  
  
}

}