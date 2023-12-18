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

template void DenseGradientStep<PTFF, long, double>(DENSE_DOUBLE& parameter, DENSE_DOUBLE& gradient, double lr);
template void BiasGradientStep<PTFF, long, double>(std::vector<double>* parameter, DenseMatrix<double>& gradient, double lr);

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
  int offset = myrank * num_diag_elements;
  for(int i = 0; i < num_diag_elements; i++){
    out->at(i+offset) += dL_dw.getValues()->at(i + i * dL_dw.getLocalCols());
  }
}

void WDerivativeUpdate(std::shared_ptr<CommGrid> comm_grid, std::vector<double>* acc_grads, std::vector<double>* out, double lr){
  int num_diag_elements = acc_grads->size();
  int diag_comm_size; MPI_Comm_size(comm_grid->GetDiagWorld(), &diag_comm_size);
  int local_diag_el = num_diag_elements / diag_comm_size;
  int global_rows, global_cols;

  if(acc_grads->size() != out->size()){
    throw std::invalid_argument("acc_grads vector needs does not have the correct dim (DerivativeFunctions.cpp, WDerivativeUpdate)");
  }

  // First perform local update of the w vector (only the diagonal elements)
  int myrank = comm_grid->GetDiagRank();
  if(myrank != MPI_PROC_NULL){
    int offset = myrank * local_diag_el;
    for(int i = offset; i < offset+num_diag_elements; i++){
      out->at(i) = out->at(i) + (-lr * acc_grads->at(i));
    }

    // Now distribute the updated diagonal elements to the diagonal processes
    std::vector<int> recvcounts = std::vector<int>(comm_grid->GetGridRows(), local_diag_el);
    std::vector<int> offsetv = std::vector<int>(comm_grid->GetGridRows());
    for (int i = 0; i < comm_grid->GetGridRows(); i++){
        offsetv[i] = i * local_diag_el;
    }
    recvcounts[recvcounts.size()-1] = num_diag_elements - offsetv[recvcounts.size()-1];
    MPI_Allgatherv(MPI_IN_PLACE, num_diag_elements, MPI_DOUBLE, out->data(), recvcounts.data(), offsetv.data(), MPI_DOUBLE, comm_grid->GetDiagWorld());
  }
  // Now broadcast the updated diagonal elements to all processes in the diagonal
  int mycol = comm_grid->GetRankInProcCol();
  MPI_Bcast(out->data(), out->size(), MPI_DOUBLE, mycol, comm_grid->GetRowWorld());

  
}


template<typename SR, typename IT, typename NT>
void DenseGradientStep(DenseMatrix<NT>& parameter, DenseMatrix<NT>& gradient, double lr){
    size_t rows = parameter.getLocalRows(); 
    size_t cols = parameter.getLocalCols();
    if (rows != gradient.getLocalRows() || cols != gradient.getLocalCols()) {
        throw std::invalid_argument( "DIMENSIONS DON'T MATCH" );        
    }
    auto dense_parameter = parameter.getValues();
    auto dense_gradient = gradient.getValues();
    #pragma omp parallel for
    for(int i = 0; i < rows * cols; i++){
        dense_parameter->at(i) = SR::add(dense_parameter->at(i), SR::multiply(static_cast<NT>(-lr), dense_gradient->at(i)));
    }
}

template<typename SR, typename IT, typename NT>
void BiasGradientStep(std::vector<double>* parameter, DenseMatrix<NT>& gradient, double lr){
  // The gradient of the bias is the sum of over all gradients in the column of the gradient matrix dL/dG2
  // First reduce the gradient matrix to a distributed vector over the columns of commGrid
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  int cols = gradient.getLocalCols();
  int rows = gradient.getLocalRows();
  std::vector<NT> local_sum(cols, 0.0);
  //TODO: Parallelize
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      local_sum.at(j) += gradient.getValues()->at(i*cols + j);
    }
  }

  // Now reduce the local sums to the root process of each column
  std::shared_ptr<CommGrid> comm_grid = gradient.getCommGrid();
  int mycolrank = comm_grid->GetRankInProcCol();
  std::vector<NT> global_sum(cols, 0.0);
  if(mycolrank == 0){
    MPI_Reduce(MPI_IN_PLACE, &local_sum[0], cols, MPIType<NT>(), MPI_SUM, 0, comm_grid->GetColWorld());
  } else{
    MPI_Reduce(&local_sum[0], &local_sum[0], cols, MPIType<NT>(), MPI_SUM, 0, comm_grid->GetColWorld());
  }

  // Update the part of the bias vector that is stored on this process 
  if(mycolrank == 0){
    int myrowrank = comm_grid->GetRankInProcRow();
    int total_length = parameter->size();
    int local_cols = total_length / comm_grid->GetGridCols();
    std::vector<int> recvcounts = std::vector<int>(comm_grid->GetGridRows(), local_cols);
    std::vector<int> offset = std::vector<int>(comm_grid->GetGridCols());
    for (int i = 0; i < comm_grid->GetGridCols(); i++){
      //TODO: Find out what cols actually has to be here (VERY IMPORTANT) this will break for the last node
        offset[i] = i * local_cols;
    }
    recvcounts[recvcounts.size()-1] = total_length - offset[recvcounts.size()-1];
    for(int i = offset[myrowrank]; i < offset[myrowrank] + recvcounts[myrowrank]; i++){
      parameter->at(i) = SR::add(parameter->at(i), SR::multiply(static_cast<NT>(-lr), local_sum.at(i - offset[myrowrank])));
    }

    MPI_Allgatherv(MPI_IN_PLACE, cols, MPIType<NT>(), parameter->data(), recvcounts.data(), offset.data(), MPIType<NT>(), comm_grid->GetRowWorld());
  }

  // What is left is to broadcast the updated bias vector to all processes in the column from the column root
  MPI_Bcast(parameter->data(), parameter->size(), MPIType<NT>(), 0, comm_grid->GetColWorld());
    
}
