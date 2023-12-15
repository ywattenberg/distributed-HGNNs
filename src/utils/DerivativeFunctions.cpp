#include "DerivativeFunctions.h"

#include <iostream>
#include <cmath>
#include <vector>

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/ParFriends.h"

#include "../utils/DenseMatrix.h"
#include "../utils/configParse.h"
#include "../utils/parDenseGEMM.h"
#include "../utils/DenseMatrix.h"

using namespace combblas;

typedef SpDCCols <int64_t, double> DCCols;
typedef SpParMat <int64_t, double, DCCols> MPI_DCCols;
typedef SpParMat<int64_t, double, SpDCCols<int64_t, double>> SPMAT_DOUBLE;
typedef DenseMatrix<double> DENSE_DOUBLE;

typedef PlusTimesSRing<double, double> PTFF;

template DenseGradientStep<PTFF, int, double>(DENSE_DOUBLE* parameter, DENSE_DOUBLE* gradient, double lr);
template VecGradientStep(std::vector<double>* parameter, DenseMatrix<double>* gradient, double lr);

void WDerivativeLocalAdd(DENSE_DOUBLE& dL_dw, std::vector<double>* out){
  //If this node is part of the diagonal of the commgrid, then we can just add the diagonal elements to the out vector
  CommGrid* comm_grid = dL_dw.getCommGrid();
  int myrank = comm_grid->GetDiagRank();
  if(myrank ==MPI_PROC_NULL){
    return;
  }
  int num_diag_elements = std::min(dL_dw.getLocalRows(), dL_dw.getLocalCols());

  if(out->size() != num_diag_elements){
    throw std::invalid_argument("out vector needs does not have the correct dim (DerivativeFunctions.cpp, WDerivativeLocalAdd)");
  }
  for(int i = 0; i < num_diag_elements; i++){
    out->at(i) += dL_dw.getValues()->at(i + i * dL_dw.getLocalCols());
  }
}

void WDerivativeAccumulation(DENSE_DOUBLE& dL_dw, std::vector<double>* out){
  int num_diag_elements = std::min(dL_dw.getLocalRows(), dL_dw.getLocalCols());
  int global_rows, global_cols;
  global_rows = dL_dw.getnrow();
  global_cols = dL_dw.getncol();
  if(std::min(global_cols, global_rows) == out->size()){
    throw std::invalid_argument("out vector needs does not have the correct dim (DerivativeFunctions.cpp, WDerivativeAccumulation)");
  }
  CommGrid* comm_grid = dL_dw.getCommGrid();
  std::vector<int> recvcounts = std::vector<int>(comm_grid->GetGridRows(), num_diag_elements);
  std::vector<int> offset = std::vector<int>(comm_grid->GetGridRows());
  for (int i = 0; i < comm_grid->GetGridRows(); i++){
      offset[i] = i * num_diag_elements;
  }
  std::vector<double> local_diag = std::vector<double>(num_diag_elements);
  for (int i = 0; i < num_diag_elements; i++){
      local_diag[i] = dL_dw.getValues()->at(i + i * dL_dw.getLocalCols());
  }
  MPI_Gatherv(local_diag.data(), num_diag_elements, MPI_DOUBLE, out->data(), recvcounts.data(), offset.data(), MPI_DOUBLE, 0, comm_grid->GetWorld());   
}


// TODO: Parallelize 
template<typename SR, typename IT, typename NT>
void DenseGradientStep(DenseMatrix<NT>* parameter, DenseMatrix<NT>* gradient, double lr){
    size_t rows = parameter->getLocalRows(); 
    size_t cols = parameter->getLocalCols();
    if (rows != gradient->getLocalRows() || cols != gradient->getLocalCols()) {
        throw std::invalid_argument( "DIMENSIONS DON'T MATCH" );        
    }
    auto dense_parameter = parameter->getValues();
    auto dense_gradient = gradient->getValues();
    for(int i = 0; i < rows * cols; i++){
        dense_parameter->at(i) = SR::add(dense_parameter->at(i), SR::multiply(static_cast<NT>(-lr), dense_gradient->at(i)));
    }
}
//TODO: FIX when we know how FullyDistVec is implemented and Parallelize
template<typename SR, typename IT, typename NT>
void VecGradientStep(std::vector<double>* parameter, DenseMatrix<NT>* gradient, double lr){
    int rows = gradient->getLocalRows(); 
    int cols = gradient->getLocalCols();
    if (rows != gradient->getLocalRows() || cols != gradient->getLocalCols()) {
        throw std::invalid_argument( "DIMENSIONS DON'T MATCH" );        
    }
    auto dense_gradient = gradient->getValues();
    for(int i = 0; i < rows; i++){
        //Accumulate the gradient over the columns and adjust the parameter vector accordingly
        NT avg = static_cast<NT>(0);
        for(int j = 0; j < cols; j++){
            avg = SR::add(avg, dense_gradient->at(j + i * cols));
        }
        avg = SR::multiply(avg, static_cast<NT>(1.0/cols));
        parameter->at(i) = SR::add(parameter->at(i), SR::multiply(static_cast<NT>(-lr), avg));
    }
    
}