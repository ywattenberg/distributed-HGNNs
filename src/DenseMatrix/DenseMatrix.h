#ifndef _DENSE_MATRIX_H_
#define _DENSE_MATRIX_H_


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


namespace combblas{

  


  template<class NT>
  class DenseMatrix 
  {
    public:
      //Constructors
      DenseMatrix<NT>();
      DenseMatrix<NT>(int rows, int cols, std::vector<NT> * values, std::shared_ptr<CommGrid> grid);
      DenseMatrix<NT>(int rows, int cols, std::shared_ptr<CommGrid> grid);
      DenseMatrix<NT>(int rows, int cols, std::vector<NT> *values, MPI_Comm world);
      DenseMatrix<NT>(int rows, int cols, MPI_Comm world);
      DenseMatrix(DenseMatrix<NT> &A, bool transpose=false); //Transpose Constructor or copy constructor

      void printLocalMatrix();

      std::vector<NT>* getValues();
      void setValues(std::vector<NT>* vals);
      void push_back(NT val);

      int getLocalRows() const;
      int getLocalCols() const;
      void setLocalRows(int rows);
      void setLocalCols(int cols);
      void GetPlaceInGlobalGrid(int &roffset, int &coffset);

      std::shared_ptr<CommGrid> getCommGrid();
      inline MPI_Comm getCommWorld();

      int getnrow();
      int getncol();
      void GetPlaceInGlobalGrid(int &roffset, int &coffset) const;

      void ParallelReadDMM (const std::string & filename, bool onebased);
      void addBiasLocally(std::vector<NT>* bias);
      void clear();

      std::vector<NT> * getRowSum();

      inline int getOwner(int nrows, int ncols, int row, int col, int grid_len) const;
      

    private:
      int localRows;
      int localCols;
      int globalRows = -1;
      int globalCols = -1;
      std::vector<NT> *values;
      std::shared_ptr<CommGrid> commGrid;

  };

  template<class NT>
  DenseMatrix<NT>::DenseMatrix(): values(nullptr), localRows(-1), localCols(-1), commGrid(nullptr){}

  template<class NT>
  DenseMatrix<NT>::DenseMatrix(int rows, int cols, std::vector<NT> * values, std::shared_ptr<CommGrid> grid): values(values), localRows(rows), localCols(cols)
  {
    commGrid = grid;
  }

  template<class NT>
  DenseMatrix<NT>::DenseMatrix(int rows, int cols, std::shared_ptr<CommGrid> grid): localRows(rows), localCols(cols), values(nullptr)
  {
    commGrid = grid;
  }

  template<class NT>
  DenseMatrix<NT>::DenseMatrix(int rows, int cols, std::vector<NT> *values, MPI_Comm world): values(values), localRows(rows), localCols(cols)
  {
    commGrid.reset(new CommGrid(MPI_COMM_WORLD, rows, cols));
  }

  template<class NT>
  inline MPI_Comm DenseMatrix<NT>::getCommWorld(){return commGrid->GetWorld();}

  template<class NT>
  inline int DenseMatrix<NT>::getOwner(int nrows, int ncols, int row, int col, int grid_len) const
  {
      int rows_per_proc = nrows / grid_len;
      int cols_per_proc = ncols / grid_len;
      int rowid = row / rows_per_proc;
      rowid = std::min(rowid, grid_len - 1);
      int colid = col / cols_per_proc;
      colid = std::min(colid, grid_len - 1);
      int rank = rowid * grid_len + colid;
      // std::cout << "row " << row << " col " << col << " is owned by rank " << rank << std::endl;
      return rank;
  }

  template <class NT>
  void processLines(std::vector<std::string>& lines, int type, std::vector<NT>& vals, int myrank);
}

#endif