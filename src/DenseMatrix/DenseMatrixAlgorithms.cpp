#include "DenseMatrixAlgorithms.h"

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


namespace combblas{

typedef SpDCCols <int64_t, double> DCCols;
typedef SpParMat <int64_t, double, DCCols> MPI_DCCols;
typedef SpParMat<int64_t, double, SpDCCols<int64_t, double>> SPMAT_DOUBLE;
typedef DenseMatrix<double> DENSE_DOUBLE;
typedef PlusTimesSRing<double, double> PTFF;

template void BCastMatrixDense<long, double>(MPI_Comm & comm1d, std::vector<double> * values, std::vector<long> &essentials, int sendingRank);
template void BCastMatrixDense<int, double>(MPI_Comm & comm1d, std::vector<double> * values, std::vector<int> &essentials, int sendingRank);

template<typename IT, typename NT>	
void BCastMatrixDense(MPI_Comm & comm1d, std::vector<NT> * values, std::vector<IT> &essentials, int sendingRank)
{
  int myrank;
  MPI_Comm_rank(comm1d, &myrank);

  MPI_Bcast(essentials.data(), essentials.size(), MPIType<IT>(), sendingRank, comm1d);

  if (myrank != sendingRank){
    values->resize(essentials[0]);
  }

  MPI_Bcast(values->data(), essentials[0], MPIType<NT>(), sendingRank, comm1d);


}


template void blockDenseSparse<PTFF, int64_t, double, DCCols>(size_t dense_rows, size_t dense_cols, std::vector<double>* dense_A, DCCols* sparse_B, std::vector<double> * outValues);

template<typename SR, typename IT, typename NT, typename DER>
void blockDenseSparse(size_t dense_rows, size_t dense_cols, std::vector<NT>* dense_A, DER* sparse_B, std::vector<NT> * outValues){
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  
  IT nnz = sparse_B->getnnz();
  IT cols_spars = sparse_B->getncol();
  IT rows_spars = sparse_B->getnrow();

  if (dense_cols != rows_spars) {
    throw std::invalid_argument( "DIMENSIONS DON'T MATCH" );        
  }

  if (nnz == 0){
    return;
  }

  Dcsc<IT, NT>* Bdcsc = sparse_B->GetDCSC();
  #pragma omp parallel for
  for(size_t i = 0; i < dense_rows; i++){
    for (size_t j = 0; j < Bdcsc->nzc; j++){
      IT col = Bdcsc->jc[j];
      size_t nnzInCol = Bdcsc->cp[j+1] - Bdcsc->cp[j];
      for(size_t k =0; k < nnzInCol; k++){
        IT sparseRow = Bdcsc->ir[Bdcsc->cp[j]+ k];
        NT elem = Bdcsc->numx[Bdcsc->cp[j]+ k];
        outValues->at(i * cols_spars + col) += SR::multiply(dense_A->at(i * dense_cols + sparseRow), elem);
      }
    }
  }
}


template void blockSparseDense<PTFF, int64_t, double, DCCols>(size_t dense_rows, size_t dense_cols, DCCols* sparse_B, std::vector<double>* dense_A, std::vector<double> * outValues);

template<typename SR, typename IT, typename NT, typename DER>
void blockSparseDense(size_t dense_rows, size_t dense_cols, DER* sparse_B, std::vector<NT>* dense_A, std::vector<NT> * outValues){
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  
  IT nnz = sparse_B->getnnz();
  IT cols_spars = sparse_B->getncol();
  IT rows_spars = sparse_B->getnrow();

  if (cols_spars != dense_rows) {
    throw std::invalid_argument( "DIMENSIONS DON'T MATCH: ");        
  }

  if (nnz == 0){
    return;
  }

  Dcsc<IT, NT>* Bdcsc = sparse_B->GetDCSC();

  for (size_t i = 0; i < Bdcsc->nzc; i++){
    IT Col = Bdcsc->jc[i];
    size_t nnzInCol = Bdcsc->cp[i+1] - Bdcsc->cp[i];
    for (size_t j = 0; j < nnzInCol; j++){
      IT sparseRow = Bdcsc->ir[Bdcsc->cp[i]+ j];
      NT elem = Bdcsc->numx[Bdcsc->cp[i]+ j];
      #pragma omp parallel for
      for (size_t k = 0; k < dense_cols; k++){
        outValues->at(sparseRow * dense_cols + k) += SR::multiply(elem, dense_A->at(Col * dense_cols + k));
      }
    }
  }
}

template DenseMatrix<double> DenseDenseAdd<PTFF, double>(DenseMatrix<double> &A, DenseMatrix<double> &B);

template<typename SR, typename NT>
DenseMatrix<NT> DenseDenseAdd(DenseMatrix<NT> &A, DenseMatrix<NT> &B){
  size_t rows = A.getLocalRows(); 
  size_t cols = A.getLocalCols();
  if (rows != B.getLocalRows() || cols != B.getLocalCols()) {
    throw std::invalid_argument( "DIMENSIONS DON'T MATCH" );        
  }
  std::vector<NT>* out = new std::vector<NT>(rows * cols);
  auto dense_A = A.getValues();
  auto dense_B = B.getValues();
  
  #pragma omp parallel for
  for(int i = 0; i < rows * cols; i++){
    out->at(i) = SR::add(dense_A->at(i), dense_B->at(i));
  }

  return DenseMatrix<NT>(rows, cols, out, A.getCommGrid()); 
}


template DenseMatrix<double> DenseVecAdd<PTFF, double>(DenseMatrix<double> &A, std::vector<double>* B);

template<typename SR, typename NT>
DenseMatrix<NT> DenseVecAdd(DenseMatrix<NT> &A, std::vector<NT>* B){
  int rows = A.getLocalRows();
  int cols = A.getLocalCols();
  auto commGrid = A.getCommGrid();

  //Find location of local rows in overall grid
  int rowDense = commGrid->GetGridRows();
  int colDense = commGrid->GetGridCols();
  int rankAinRow = commGrid->GetRankInProcRow();

  std::vector<NT>* out = new std::vector<NT>(rows * cols);
  // Add corresponding entries of B to A (offset by global pos)
  for (int i = 0; i < rows; i++){
    for (int j = 0; j < cols; j++){
      out->at(i * cols + j) = SR::add(A.getValues()->at(i * cols + j), B->at(i + rankAinRow * rows));
    }
  }

  return DenseMatrix<NT>(rows, cols, out, commGrid);
}


template DenseMatrix<double> DenseReLU<double>(DenseMatrix<double> &A);

template<typename NT>
DenseMatrix<NT> DenseReLU(DenseMatrix<NT> &A){
  size_t rows = A.getLocalRows(); 
  size_t cols = A.getLocalCols();
  std::vector<NT>* out = new std::vector<NT>(rows * cols);
  auto dense_A = A.getValues();
  #pragma omp parallel for
  for(int i = 0; i < rows * cols; i++){
    out->at(i) = dense_A->at(i) > 0 ? dense_A->at(i) : static_cast<NT>(0.0);
  }
  return DenseMatrix<NT>(rows, cols, out, A.getCommGrid());
}


template DenseMatrix<double> DerivativeDenseReLU<double>(DenseMatrix<double> &A);
template<typename NT>

DenseMatrix<NT> DerivativeDenseReLU(DenseMatrix<NT> &A){
  size_t rows = A.getLocalRows(); 
  size_t cols = A.getLocalCols();
  std::vector<NT>* out = new std::vector<NT>(rows * cols);
  auto dense_A = A.getValues();
  #pragma omp parallel for
  for(int i = 0; i < rows * cols; i++){
    out->at(i) = dense_A->at(i) > 0 ?  static_cast<NT>(1.0) : static_cast<NT>(0.0);
  }
  return DenseMatrix<NT>(rows, cols, out, A.getCommGrid());
}


template void blockDenseDense<PTFF, double>(size_t rowsA, size_t colsA, size_t rowsB, size_t colsB, std::vector<double>* dense_A, std::vector<double>* dense_B, std::vector<double>* outValues);

template<typename SR, typename NT>
void blockDenseDense(size_t rowsA, size_t colsA, size_t rowsB, size_t colsB, std::vector<NT>* dense_A, std::vector<NT>* dense_B, std::vector<NT>* outValues)
{
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  if (colsA != rowsB) {
    throw std::invalid_argument("DIMENSIONS DON'T MATCH");
  }
    #pragma omp parallel for
    for (size_t i = 0; i < rowsA; i++){
      for (size_t j = 0; j < colsB; j++){
        for (size_t k = 0; k < colsA; k++){
        outValues->at(i * colsB + j) += SR::multiply(dense_A->at(i * colsA + k), dense_B->at(k* colsB + j));
      }
    }
  }
}


template DenseMatrix<double> DenseElementWiseMult<PTFF, double>(DenseMatrix<double> &A, DenseMatrix<double> &B);

template<typename SR, typename NT>
DenseMatrix<NT> DenseElementWiseMult(DenseMatrix<NT> &A, DenseMatrix<NT> &B)
{
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  int localRowsA = A.getLocalRows();
  int localColsA = A.getLocalCols();
  int localRowsB = B.getLocalRows();
  int localColsB = B.getLocalCols();
  if(localRowsA != localRowsB) {
    throw std::invalid_argument("DIMENSIONS DON'T MATCH localRowsA != localRowsB at DenseTransDenseMult");
  }
  if(localColsA != localColsB) {
    throw std::invalid_argument("DIMENSIONS DON'T MATCH localColsA != localColsB at DenseTransDenseMult");
  }

  std::vector<NT>* res = new std::vector<NT>(localRowsA * localColsA);
  #pragma omp parallel for
  for(int i = 0; i < localRowsA * localColsA; i++) {
    res->at(i) = SR::multiply(A.getValues()->at(i), B.getValues()->at(i));
  }
  return DenseMatrix<NT>(localColsA, localColsB, res, A.getCommGrid());  
}
}
