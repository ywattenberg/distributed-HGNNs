#include "LossFn.h"

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/ParFriends.h"

#include "../utils/DenseMatrix.h"
#include "../utils/configParse.h"
#include "../utils/parDenseGEMM.h"
#include "../utils/DenseMatrix.h"

using namespace combblas;


// In this function, we only calculate the loss and not the derivative.
// if mean is true, then we calculate the mean of the loss
// else individual loss values are returned
// TODO: Parallelize this function
template <typename SR, typename NT>
DenseMatrix<NT> CrossEntropyLoss(DenseMatrix<NT> &pred, const std::vector<NT>* target, bool mean)
{
  // Calculate the Cross Entropy Loss without averaging over the graph
  // We assume that the pred matrix are input logits and not probabilities
  // For definition of Cross Entropy Loss see: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
  // Where we don't have a weight or ignore_index parameter
  std::vector<NT>* predictions = pred.getValues();
  int local_cols = pred.getLocalCols();
  int local_rows = pred.getLocalRows();

  std::vector<NT> max_val(local_rows, numeric_limits<NT>::min());
  for(int i = 0; i < local_rows; i++){
    for(int j = 0; j < local_cols; j++){
      max_val.at(i) = max(max_val.at(i), predictions->at(i*local_cols + j));
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &max_val[0], local_rows, MPI_DOUBLE, MPI_MAX, pred->getCommGrid()->GetRowWorld());

  int myrank = pred->getCommGrid()->GetRankInProcRow();
  for(int i = 0; i < local_rows; i++){
    std::cout << "Rank: " << myrank << "At location: " << i  << " Max Val: " << max_val.at(i) << std::endl;
  }

  


   
}



template <typename SR, typename NT>
DenseMatrix<NT> DerivativeCrossEntropyLoss(DenseMatrix<NT> &pred, const std::vector<NT>* target)
{
    DenseMatrix<NT> loss(pred.getnrow(), pred.getncol());
    loss.Fill(0.0);
    int64_t nrow = pred.getnrow();
    int64_t ncol = pred.getncol();
    NT* pred_data = pred.GetMatrix();
    NT* loss_data = loss.GetMatrix();
    for(int64_t i = 0; i < nrow; i++)
    {
        for(int64_t j = 0; j < ncol; j++)
        {
            loss_data[i*ncol + j] = -1.0 / pred_data[i*ncol + j];
        }
    }
    return loss;
}