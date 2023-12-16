#include "../utils/DerivativeFunctions.h"

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

template void DenseGradientStep<PTFF, int, double>(DENSE_DOUBLE* parameter, DENSE_DOUBLE* gradient, double lr);
template void BiasGradientStep<PTFF, int, double>(std::vector<double>* parameter, DenseMatrix<double>& gradient, double lr);

void WDerivativeLocalAdd(DENSE_DOUBLE& dL_dw, std::vector<double>* out){
  //If this node is part of the diagonal of the commgrid, then we can just add the diagonal elements to the out vector
  int myrank = dL_dw.getCommGrid()->GetDiagRank();
  if(myrank == MPI_PROC_NULL){
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
  std::shared_ptr<CommGrid> comm_grid = dL_dw.getCommGrid();
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
void DenseGradientStep(DenseMatrix<NT>& parameter, DenseMatrix<NT>& gradient, double lr){
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

template<typename SR, typename IT, typename NT>
void BiasGradientStep(std::vector<double>* parameter, DenseMatrix<NT>& gradient, double lr){
  // The gradient of the bias is the sum of over all gradients in the column of the gradient matrix dL/dG2
  // First reduce the gradient matrix to a distributed vector over the columns of commGrid

  int cols = gradient->getLocalCols();
  int rows = gradient->getLocalRows();
  std::vector<double> local_sum(cols, 0.0);
  //TODO: Parallelize
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      local_sum.at(j) += gradient->getValues()->at(i*cols + j);
    }
  }
  
  // Now reduce the local sums to the root process of each column
  std::shared_ptr<CommGrid> comm_grid = gradient->getCommGrid();
  int mycolrank = comm_grid->GetRankInProcCol();
  if(mycolrank == 0){
    MPI_Reduce(MPI_IN_PLACE, local_sum.data(), cols, MPI_DOUBLE, MPI_SUM, 0, comm_grid->GetColWorld());
  }else{
    MPI_Reduce(local_sum.data(), local_sum.data(), cols, MPI_DOUBLE, MPI_SUM, 0, comm_grid->GetColWorld());
  }

  // Update the part of the bias vector that is stored on this process 
  if(mycolrank == 0){
    int myrowrank = comm_grid->GetRankInProcRow();
    int total_length = parameter->size();
    std::vector<int> recvcounts = std::vector<int>(comm_grid->GetGridRows(), cols);
    std::vector<int> offset = std::vector<int>(comm_grid->GetGridCols());
    for (int i = 0; i < comm_grid->GetGridCols(); i++){
      //TODO: Find out what cols actually has to be here (VERY IMPORTANT) this will break for the last node
        offset[i] = i * cols;
    }
    recvcounts[recvcounts.size()-1] = total_length - offset[revcounts.size()-1];
    for(int i = offset[myrowrank]; i < offset[myrowrank] + recvcounts[myrowrank]; i++){
      parameter->at(i) = SR::add(parameter->at(i), SR::multiply(static_cast<NT>(-lr), local_sum.at(i)));
    }
    MPI_Allgatherv(MPI_IN_PLACE, cols, MPIType<NT>(), parameter->data(), recvcounts.data(), offset.data(), MPIType<NT>(), comm_grid->GetRowWorld());
  }

  // What is left is to broadcast the updated bias vector to all processes in the column from the column root
  MPI_Bcast(parameter->data(), parameter->size(), MPIType<NT>(), 0, comm_grid->GetColWorld());
    
}
