#include "DenseMatrix.h"

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <omp.h>
#include <cblas.h>
// #include <mkl.h>

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/SpParMat.h"
#include "CombBLAS/ParFriends.h"
#include "CombBLAS/SpParHelper.h"
#include "CombBLAS/SpDCCols.h"

namespace combblas {

  template class DenseMatrix<double>;

  template<typename NT>
  void DenseMatrix<NT>::printLocalMatrix(){
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    std::cout << "local matrix A:" << std::endl;
    for (int i = 0; i < this->localRows * this->localCols; i++) {
        std::cout << "rank " << myrank << " local matrix A[" << i << "] = " << this->values->at(i) << std::endl;
    }
  }


  template<typename NT>
  std::vector<NT>* DenseMatrix<NT>::getValues() {return values;}


  template<typename NT>
  void DenseMatrix<NT>::setValues(std::vector<NT>* vals) {values = vals;}


  template<typename NT>
  void DenseMatrix<NT>::push_back(NT val) {values->push_back(val);}


  template<typename NT>
  int DenseMatrix<NT>::getLocalRows() const {return localRows;}


  template<typename NT>
  int DenseMatrix<NT>::getLocalCols() const {return localCols;}


  template<typename NT>
  void DenseMatrix<NT>::setLocalRows(int rows) {localRows = rows;}


  template<typename NT>
  void DenseMatrix<NT>::setLocalCols(int cols) {localCols = cols;}


  template<typename NT>
  std::shared_ptr<CommGrid> DenseMatrix<NT>::getCommGrid() {return commGrid;}

  // transpose Constructor
  template<typename NT>
  DenseMatrix<NT>::DenseMatrix(DenseMatrix<NT> &A, bool transpose){

    commGrid = A.getCommGrid();
    int rankInRow = commGrid->GetRankInProcRow();
    int rankInCol = commGrid->GetRankInProcCol();
    int rowsA = A.getLocalRows();
    int colsA = A.getLocalCols();

    if(!transpose){
      this->values = new std::vector<NT>(rowsA*colsA);
      this->localRows = rowsA;
      this->localCols = colsA;

      for(int i = 0; i < rowsA*colsA; i++){
        this->values->at(i) = A.getValues()->at(i);
      } 
      return;
    }

    int rowsNew, colsNew;
    std::vector<NT> * buffer = new std::vector<NT>();
    values = new std::vector<NT>();

    if (rankInRow != rankInCol){
      int diagNeighbour = commGrid->GetComplementRank();
      
      MPI_Status status;
      MPI_Sendrecv(&rowsA, 1, MPI_INT, diagNeighbour, 0, &colsNew, 1, MPI_INT, diagNeighbour, 0, commGrid->GetWorld(), &status);
      MPI_Sendrecv(&colsA, 1, MPI_INT, diagNeighbour, 0, &rowsNew, 1, MPI_INT, diagNeighbour, 0, commGrid->GetWorld(), &status);

      buffer->resize(rowsNew * colsNew);
      localRows = rowsNew;
      localCols = colsNew;

      MPI_Sendrecv(A.getValues()->data(), rowsA * colsA, MPIType<NT>(), diagNeighbour, 0, buffer->data(), rowsNew * colsNew, MPIType<NT>(), diagNeighbour, 0, commGrid->GetWorld(), &status);
    } else {
      localRows = colsA;
      localCols = rowsA;

      // buffer->resize(rowsA * colsA);
      std::copy(A.getValues()->begin(), A.getValues()->end(), std::back_inserter(*buffer));
      // std::copy(A.getValues()->begin(), A.getValues()->end(), buffer->begin());
    }

    values->resize(localRows * localCols);
    // transpose locally
    for (int i = 0; i < localCols; i++){
      for (int j = 0; j < localRows; j++){
        values->at(j * localCols + i) = buffer->at(i*localRows + j);
      }
    }

    delete buffer;
  }


  template<typename NT>
  void DenseMatrix<NT>::clear()
  {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    localRows = -1;
    localCols = -1;
    globalRows = -1;
    globalCols = -1;
    if (this->values != nullptr) {
      delete this->values;
      this->values = nullptr;
    }
  }
  

  template<typename NT>
  int DenseMatrix<NT>::getnrow()
  {
    if(this->globalRows > -1)return this->globalRows;

    int totalrows = 0;  
    MPI_Allreduce( &localRows, &totalrows, 1, MPI_INT, MPI_SUM, commGrid->GetColWorld());
    this->globalRows = totalrows;
    return totalrows;  
  }


  template<typename NT>
  int DenseMatrix<NT>::getncol()
  {
    if(this->globalCols > -1)return this->globalCols;
    int totalcols = 0; 
    MPI_Allreduce( &localCols, &totalcols, 1, MPI_INT, MPI_SUM, commGrid->GetRowWorld());
    this->globalCols = totalcols;
    return totalcols;  
  }


  template<typename NT>
  void DenseMatrix<NT>::GetPlaceInGlobalGrid(int &roffset, int &coffset){
    int total_rows = getnrow();
    int total_cols = getncol();

    int proc_rows = commGrid->GetGridRows();
    int proc_cols = commGrid->GetGridCols();

    int rows_perproc = total_rows / proc_rows;
    int cols_perproc = total_cols / proc_cols;

    roffset = commGrid->GetRankInProcCol()*rows_perproc;
    coffset = commGrid->GetRankInProcRow()*cols_perproc;
  }
  

  template<typename NT>
  void DenseMatrix<NT>::addBiasLocally(std::vector<NT>* bias)
  {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int localRows = getLocalRows();
    int localCols = getLocalCols();

    int roffset = 0;
    int coffset = 0;
    GetPlaceInGlobalGrid(roffset, coffset);

    for (int i = 0; i < localRows; i++){
      for (int j = 0; j < localCols; j++){
        // std::cout << "bias elem " <<  bias->at(coffset + j) << std::endl;
        // std::cout << "value element: " << values->at(i*localCols + j) << std::endl;
        values->at(i*localCols + j) += bias->at(coffset + j);
      }
    }
  }


  template<typename NT>
  std::vector<NT> * DenseMatrix<NT>::getRowSum()
  {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    std::vector<NT> rowSums = std::vector<NT>(localRows);

    for (int i = 0; i < localRows; i++){
      for (int j = 0; j < localCols; j++){
        rowSums.at(i) = rowSums.at(i) + values->at(i*localCols + j);
      }
    }

    MPI_Allreduce(MPI_IN_PLACE, rowSums.data(), localRows, MPIType<NT>(), MPI_SUM, commGrid->GetRowWorld());

    int totalRows = getnrow();
    int gridRows = commGrid->GetGridRows();
    std::vector<int> displacements = std::vector<int>(gridRows);
    std::vector<int> sizes = std::vector<int>(gridRows, localRows);
    

    for (int i = 0; i < gridRows; i++){
      displacements.at(i) = i*localRows;
    }

    sizes.at(gridRows-1) = totalRows - displacements.at(gridRows-1);

    std::vector<NT> * allRowSums = new std::vector<NT>(totalRows);
    MPI_Gatherv(rowSums.data(), localRows, MPIType<NT>(), allRowSums->data(), sizes.data(), displacements.data(), MPIType<NT>(), 0, commGrid->GetColWorld());
    MPI_Bcast(allRowSums->data(), totalRows, MPIType<NT>(), 0, commGrid->GetColWorld());

    return allRowSums;

  }
}