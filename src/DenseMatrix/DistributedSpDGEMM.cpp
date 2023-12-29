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
  
  template DenseMatrix<double> DenseSpMult<PTFF, int64_t, double, DCCols>(DenseMatrix<double> &A, SpParMat<int64_t, double, SpDCCols<int64_t, double>> &B);

  template<typename SR, typename IT, typename NT, typename DER>
  DenseMatrix<NT> DenseSpMult(DenseMatrix<NT> &A, SpParMat<IT, NT, DER> &B) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    int rowDense = A.getCommGrid()->GetGridRows();
    int colDense = A.getCommGrid()->GetGridCols();

    int stages, dummy;
    std::shared_ptr<CommGrid> GridC = ProductGrid((A.getCommGrid()).get(), (B.getcommgrid()).get(), stages, dummy, dummy);		
    

    IT ** BRecvSizes = SpHelper::allocate2D<IT>(DER::esscount, stages);
    SpParHelper::GetSetSizes( *(B.getSpSeq()), BRecvSizes, (B.getcommgrid())->GetColWorld());

    std::vector<IT> essentialsA(3); // saves rows, cols and total number of elements of block
    std::vector<IT> ess = std::vector<IT>();
    std::vector<NT> * out;

    int rankAinRow = A.getCommGrid()->GetRankInProcRow();
    int rankAinCol = A.getCommGrid()->GetRankInProcCol();
    int rankBinCol = B.getcommgrid()->GetRankInProcCol();

    int denseLocalRows = A.getLocalRows();
    int denseLocalCols = A.getLocalCols();

    int sparseLocalRows = B.getlocalrows();
    int sparseLocalCols = B.getlocalcols();

    std::vector<NT> * localOut = new std::vector<NT>(denseLocalRows * sparseLocalCols, 0.0);
    
    for (int i = 0; i < stages; i++){
      int sendingRank = i;
      std::vector<NT> * bufferA = new std::vector<NT>();
      DER * bufferB;

      
      if (rankAinRow == sendingRank){
        bufferA = A.getValues();

        essentialsA[1] = A.getLocalRows();
        essentialsA[2] = A.getLocalCols();
        essentialsA[0] = essentialsA[1] * essentialsA[2];
      }

      BCastMatrixDense(GridC->GetRowWorld(), bufferA, essentialsA, sendingRank);

      if(rankBinCol == sendingRank)
      {
        bufferB = B.getSpSeq();
      }
      else
      {
        ess.resize(DER::esscount);		
        for(int j=0; j< DER::esscount; ++j)	
        {
          ess[j] = BRecvSizes[j][i];
        }	
        bufferB = new DER();
      }
      
      SpParHelper::BCastMatrix<IT, NT, DER>(GridC->GetColWorld(), *bufferB, ess, sendingRank);

      blockDenseSparse<SR, IT, NT, DER>(essentialsA[1], essentialsA[2], bufferA, bufferB, localOut);

      if (rankAinRow != sendingRank){
        delete bufferA;
      }

      if (rankBinCol != sendingRank){
        delete bufferB;
      }
    }

    return DenseMatrix<NT>(denseLocalRows, sparseLocalCols, localOut, GridC);
}


  template DenseMatrix<double> SpDenseMult<PTFF, int64_t, double, DCCols>(SpParMat<int64_t, double, SpDCCols<int64_t, double>> &B, DenseMatrix<double> &A);

  template<typename SR, typename IT, typename NT, typename DER>
  DenseMatrix<NT> SpDenseMult(SpParMat<IT, NT, DER> &B, DenseMatrix<NT> &A) 
  {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    int rowDense = A.getCommGrid()->GetGridRows();
    int colDense = A.getCommGrid()->GetGridCols();

    int stages, dummy;
    std::shared_ptr<CommGrid> GridC = ProductGrid((B.getcommgrid()).get(), (A.getCommGrid()).get(), stages, dummy, dummy);		
    

    IT ** BRecvSizes = SpHelper::allocate2D<IT>(DER::esscount, stages);
    SpParHelper::GetSetSizes( *(B.getSpSeq()), BRecvSizes, (B.getcommgrid())->GetRowWorld());

    std::vector<IT> essentialsA(3); // saves rows, cols and total number of elements of block
    std::vector<IT> ess = std::vector<IT>();
    std::vector<NT> * out;

    int rankAinRow = A.getCommGrid()->GetRankInProcRow();
    int rankAinCol = A.getCommGrid()->GetRankInProcCol();
    int rankBinRow = B.getcommgrid()->GetRankInProcRow();
    int rankBinCol = B.getcommgrid()->GetRankInProcCol();

    int denseLocalRows = A.getLocalRows();
    int denseLocalCols = A.getLocalCols();

    int sparseLocalRows = B.getlocalrows();
    int sparseLocalCols = B.getlocalcols();

    std::vector<NT> * localOut = new std::vector<NT>(sparseLocalRows * denseLocalCols, 0.0);
    
    //other stages:
    for (int i = 0; i < stages; i++){
      int sendingRank = i;
      std::vector<NT> * bufferA = new std::vector<NT>();
      DER * bufferB;
      
      if (rankAinCol == sendingRank){
        bufferA = A.getValues();

        essentialsA[1] = A.getLocalRows();
        essentialsA[2] = A.getLocalCols();
        essentialsA[0] = essentialsA[1] * essentialsA[2];
      }

      BCastMatrixDense(GridC->GetColWorld(), bufferA, essentialsA, sendingRank);

      if(rankBinRow == sendingRank)
      {
        bufferB = B.getSpSeq();
      }
      else
      {
        ess.resize(DER::esscount);		
        for(int j=0; j< DER::esscount; ++j)	
        {
          ess[j] = BRecvSizes[j][i];
        }	
        bufferB = new DER();
      }
      
      SpParHelper::BCastMatrix<IT, NT, DER>(GridC->GetRowWorld(), *bufferB, ess, sendingRank);

      blockSparseDense<SR, IT, NT, DER>(essentialsA[1], essentialsA[2], bufferB, bufferA, localOut);

      if (rankAinCol != sendingRank){
        delete bufferA;
      }

      if (rankBinRow != sendingRank){
        delete bufferB;
      }
    }
    return DenseMatrix<NT>(sparseLocalRows, denseLocalCols, localOut, GridC);
  }

}