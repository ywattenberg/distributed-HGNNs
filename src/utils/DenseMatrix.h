#ifndef _DENSE_MATRIX_H_
#define _DENSE_Matrix_H_


#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <vector>

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/SpParMat.h"


namespace combblas{

template<class NT>
class DenseMatrix 
{
  public:
    std::vector<NT> *getValues() {return values;}
    int getLocalRows() {return localRows;}
    int getLocalCols() {return localCols;}
    std::shared_ptr<CommGrid> getCommGrid() {return commGrid;}

    DenseMatrix<NT>(int rows, int cols, std::vector<NT> *values, std::shared_ptr<CommGrid> grid): values(values), localRows(rows), localCols(cols)
    {
      commGrid = grid;
    }

    DenseMatrix<NT>(int rows, int cols, std::vector<NT> *values, MPI_Comm world): values(values), localRows(rows), localCols(cols)
    {
      commGrid.reset(new CommGrid(MPI_COMM_WORLD, rows, cols));
    }

    

  private:
    int localRows;
    int localCols;
    std::vector<NT> *values;
    std::shared_ptr<CommGrid> commGrid; 
};

int getSendingRankInRow(int rank, int diagOffset, int cols){
  int rowPos = rank / cols;
  return rowPos * cols + (rowPos + diagOffset) % cols;
}

int getRecvRank(int rank, int round, int cols, int size){
  int RecvRank = rank - (round * cols);
  if (RecvRank < 0){
    RecvRank = size + rank - (round * cols);
  }
  return RecvRank;
}


template<typename SR, typename IT, typename NT2, typename DER>
    DenseMatrix<NT2> * localMatrixMult(DenseMatrix<NT2> &A, SpParMat<IT, NT2, DER> &B, bool clearA=false, bool clearB=false)
    {
      DER B_elems = *B.getSpSeq();
      IT nnz = B_elems.getnnz();
      IT colsSpars = B_elems.getncol();
      IT rowsSpars = B_elems.getnrow();
      int localCols = A.getLocalCols();
      int localRows = A.getLocalRows();
      std::vector<NT2> * values = A.getValues();

      if (localCols != rowsSpars) {
        cout << "DIMENSIONS DONT MATCH";
        return -1;
      } 

      if (nnz == 0){
        std::vector<NT2> outValues =  std::vector<NT2>(localCols * localRows, 0);
        DenseMatrix<NT2> * out = new DenseMatrix<NT2>(localRows, localCols, outValues);
        return out;
      }

      Dcsc<IT, NT2>* Bdcsc = B_elems.GetDCSC();
      std::vector<NT2> outValues =  std::vector<NT2>(localCols * localRows, 0);

      for (size_t i = 0; i < B->nzc; ++i){
        IT col = Bdcsc->jc[i];
        size_t nnzInCol = Bdcsc->cp[i+1] - Bdcsc->cp[i];


        for (size_t row = 0; row < localRows; row++){ //loop over rows of the block of the dense matrix
          NT2 sum = 0;
          size_t offset = row * localCols;
          for (size_t k = Bdcsc->cp[i]; k < Bdcsc->cp[i] + nnzInCol; k++){
            IT sparseRow = Bdcsc->ir[k];
            NT2 elem = Bdcsc->numx[k];
            sum += SR::multiply(*values[offset + sparseRow], elem);
          }
          outValues[offset + col] = sum;
        }
        
      } 
      DenseMatrix<NT2> * out = new DenseMatrix<NT2>(localRows, localCols, outValues);

      return out;

      
    }


template<typename SR, typename IT, typename NT, typename DER>
DenseMatrix<NT> fox(DenseMatrix<NT> &A, SpParMat<IT, NT, DER> &B)
{
  MPI_Comm commi = A.getCommGrid().GetWorld();
  int size, myrank;
  MPI_Comm_size(commi, &size);
  MPI_Comm_rank(commi, &myrank);

  int rowDense = A.getCommGrid().GetGridRows();
  int colDense = A.getCommGrid().GetGridCols();

  int rowSparse = B.getCommGrid().GetGridRows();
  int colSparse = B.getCommGrid().GetGridCols();

  if (myrank == 0){
    if (rowDense != rowSparse || colDense != colSparse || rowDense != colDense){
      MPI_Abort(commi, 1);
    }
  }

  DER * B_elems = B.getSpSeq();
  std::vector<DenseMatrix<NT>*> results;

  // Round 0:
  std::vector<NT> bufferA;
  int size_vec;
  int sendingRank = getSendingRankInRow(myrank, 0, colDense);
  if (myrank == sendingRank){
    size_vec = A.getValues().size_vec();
    bufferA = *A.getValues();
  }
  MPI_Bcast(&size_vec, 1, MPI_INT, sendingRank, A.getCommGrid().GetRowWorld);

  bufferA.resize(size_vec);
  
  if (std::is_same<NT, double>::value){
      MPI_Bcast(bufferA.data(), size_vec, MPI_DOUBLE, sendingRank, A.getCommGrid().GetRowWorld);
  }

  DenseMatrix<NT> A_tmp = DenseMatrix<NT>(rowDense, colDense, &bufferA, A.getCommGrid);

  results.push_back(localMatrixMult(A_tmp, B));

  // other Rounds:
  std::vector<std::tuple<IT,IT,NT>> bufferB;
  for (size_t round = 1; round < colDense; round++){
    // BroadCasting A
    sendingRank = getSendingRankInRow(myrank, round, colDense);

    if (myrank == sendingRank){
      size_vec = A.getValues().size_vec();
      bufferA = *A.getValues();
    }

    MPI_Bcast(&size_vec, 1, MPI_INT, sendingRank, A.getCommGrid().GetRowWorld);
    bufferA.resize(size_vec);

    if (std::is_same<NT, double>::value){
      MPI_Bcast(bufferA.data(), size_vec, MPI_DOUBLE, sendingRank, A.getCommGrid().GetRowWorld);
    }
    
    MPI_Request send_request, recv_request;
    MPI_Status status;
    // Sending the correct B block
    int RecvRank = getRecvRank(myrank, round, colDense, size);
    // int Send_rank = 
    int sizeSparse = B_elems->getnnz();

    MPI_Isend(&sizeSparse, 1, MPI_INT, RecvRank, 0, MPI_COMM_WORLD, &send_request);
    

  }
    

}



}


#endif