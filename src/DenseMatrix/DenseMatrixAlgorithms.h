#ifndef _DENSE_MATRIX_ALGORITHMS_H_
#define _DENSE_MATRIX_ALGORITHMS_H_


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


template<typename IT, typename NT>	
extern void BCastMatrixDense(MPI_Comm & comm1d, std::vector<NT> * values, std::vector<IT> &essentials, int sendingRank);


template<typename SR, typename IT, typename NT, typename DER>
extern void blockDenseSparse(size_t dense_rows, size_t dense_cols, std::vector<NT>* dense_A, DER* sparse_B, std::vector<NT> * outValues);

template<typename SR, typename NT>
extern void blockDenseDenseTrans(size_t rowsA, size_t colsA, size_t rowsB, size_t colsB, std::vector<NT>* dense_A, std::vector<NT>* dense_B, std::vector<NT>* outValues);


template<typename SR, typename IT, typename NT, typename DER>
extern void blockSparseDense(size_t dense_rows, size_t dense_cols, DER* sparse_B, std::vector<NT>* dense_A, std::vector<NT> * outValues);


template<typename SR, typename NT>
extern DenseMatrix<NT> DenseDenseMult(DenseMatrix<NT> &A, DenseMatrix<NT> &B);

template<typename SR, typename NT>
extern DenseMatrix<NT> DenseDenseTransMult(DenseMatrix<NT> &A, DenseMatrix<NT> &B);


template<typename SR, typename NT>
extern DenseMatrix<NT> DenseTransDenseMult(DenseMatrix<NT> &A, DenseMatrix<NT> &B);


template<typename SR, typename NT>
extern DenseMatrix<NT> DenseDenseAdd(DenseMatrix<NT> &A, DenseMatrix<NT> &B);

template<typename SR, typename NT>
extern DenseMatrix<NT> DenseVecAdd(DenseMatrix<NT> &A, std::vector<NT>* B);

template<typename NT>
extern DenseMatrix<NT> DenseReLU(DenseMatrix<NT> &A);

template<typename NT>
extern DenseMatrix<NT> DerivativeDenseReLU(DenseMatrix<NT> &A);

template<typename SR, typename IT, typename NT, typename DER>
extern DenseMatrix<NT> DenseSpMult(DenseMatrix<NT> &A, SpParMat<IT, NT, DER> &B);

// computes B*A, where B is a sparse matrix and A a dense matrix
template<typename SR, typename IT, typename NT, typename DER>
extern DenseMatrix<NT> SpDenseMult(SpParMat<IT, NT, DER> &B, DenseMatrix<NT> &A);


template<typename SR, typename NT>
extern void blockDenseDense(size_t rowsA, size_t colsA, size_t rowsB, size_t colsB, std::vector<NT>* dense_A, std::vector<NT>* dense_B, std::vector<NT>* outValues);

template<typename SR, typename NT>
extern DenseMatrix<NT> DenseElementWiseMult(DenseMatrix<NT> &A, DenseMatrix<NT> &B);

}


#endif