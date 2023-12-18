#include "LossFn.h"

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
typedef PlusTimesSRing<double, double> PTFF;

template double CrossEntropyLoss<PTFF, double>(DenseMatrix<double> &pred, const std::vector<int>* target, bool sum);
template DenseMatrix<double> DerivativeCrossEntropyLoss<PTFF, double>(DenseMatrix<double> &pred, const std::vector<int>* target, bool sum_reduction);


// In this function, we only calculate the loss and not the derivative.
// if mean is true, then we calculate the mean of the loss
// else individual loss values are returned
// TODO: Parallelize this function
template <typename SR, typename NT>
NT CrossEntropyLoss(DenseMatrix<NT> &pred, const std::vector<int>* target, bool sum)
{
  // Calculate the Cross Entropy Loss without averaging over the graph
  // We assume that the pred matrix are input logits and not probabilities
  // For definition of Cross Entropy Loss see: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
  // Where we don't have a weight or ignore_index parameter
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  

  std::vector<NT>* predictions = pred.getValues();
  int local_cols = pred.getLocalCols();
  int local_rows = pred.getLocalRows();

  // Find the max value in each row for logsumexp trick
  std::vector<NT> max_val(local_rows, numeric_limits<NT>::min());
  for(int i = 0; i < local_rows; i++){
    for(int j = 0; j < local_cols; j++){
      max_val.at(i) = max(max_val.at(i), predictions->at(i*local_cols + j));
    }
  }


  MPI_Allreduce(MPI_IN_PLACE, &max_val[0], local_rows, MPIType<NT>(), MPI_MAX, pred.getCommGrid()->GetRowWorld());

  //Calculate log(sum(exp(x- max(x)))) for each row 
  std::vector local_sum(local_rows, 0.0);
  for(int i = 0; i < local_rows; i++){
    for(int j = 0; j < local_cols; j++){
      local_sum.at(i) += std::exp(SR::add(predictions->at(i*local_cols + j), -max_val.at(i)));
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &local_sum[0], local_rows, MPIType<NT>(), MPI_SUM, pred.getCommGrid()->GetRowWorld());
  MPI_Barrier(MPI_COMM_WORLD);
  // Calculate add max of row (logsumexp trick)
  for(int i = 0; i < local_rows; i++){
    local_sum.at(i) = SR::add(max_val.at(i), std::log(local_sum.at(i)));
  }

  int row_offset, col_offset;
  pred.GetPlaceInGlobalGrid(row_offset, col_offset);
  
  std::vector<NT> x_y_n(local_rows, 0.0);

  MPI_Barrier(MPI_COMM_WORLD);
  for(int i = 0; i < local_rows; i++){
    int y_n = target->at(i + row_offset);
    if(y_n >= col_offset && y_n < col_offset + local_cols){
      x_y_n.at(i) = predictions->at(i*local_cols + y_n - col_offset);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  // TODO: Gatherv might be more efficient reduce is not needed here (actually maybe not as elements are not in order)
  MPI_Allreduce(MPI_IN_PLACE, &x_y_n[0], local_rows, MPIType<NT>(), MPI_SUM, pred.getCommGrid()->GetRowWorld());
  NT loss = 0.0;
  NT total_rows = static_cast<NT>(pred.getnrow());

  for(int i = 0; i < local_rows; i++){
    // std::cout << " Rank " << myrank << " "<< x_y_n.at(i) << " + " << local_sum.at(i) << std::endl;
    loss += (- x_y_n.at(i) + local_sum.at(i))/(sum? 1.0:total_rows);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &loss, 1, MPIType<NT>(), MPI_SUM, pred.getCommGrid()->GetColWorld());
  return loss;
}



template <typename SR, typename NT>
DenseMatrix<NT> DerivativeCrossEntropyLoss(DenseMatrix<NT> &pred, const std::vector<int>* target, bool sum_reduction)
{
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  std::vector<NT>* predictions = pred.getValues();
  int local_cols = pred.getLocalCols();
  int local_rows = pred.getLocalRows();

  std::vector local_sum(local_rows, 0.0);
  for(int i = 0; i < local_rows; i++){
    for(int j = 0; j < local_cols; j++){
      local_sum.at(i) += std::exp(predictions->at(i*local_cols + j));
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &local_sum[0], local_rows, MPIType<NT>(), MPI_SUM, pred.getCommGrid()->GetRowWorld());

  int row_offset, col_offset;
  pred.GetPlaceInGlobalGrid(row_offset, col_offset);

  int total_rows = pred.getnrow();
  std::vector<NT>* out = new std::vector<NT>(local_rows * local_cols);
  for(int i = 0; i < local_rows; i++){
    for(int j = 0; j < local_cols; j++){
      // if(target->at(i + row_offset) >= col_offset && target->at(i+ row_offset) < col_offset + local_cols){
      out->at(i*local_cols + j) = std::exp(predictions->at(i* local_cols + j))/local_sum.at(i);
      out->at(i*local_cols + j) -= (col_offset + j == target->at(row_offset + i)) ? 1.0 : 0.0;
      if(!sum_reduction)out->at(i*local_cols + j) /= total_rows;
    }
  }

  return DenseMatrix<NT>(local_rows, local_cols, out, pred.getCommGrid());
}

