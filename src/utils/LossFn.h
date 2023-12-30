#ifndef CUSTOM_LOSS_FN_H
#define CUSTOM_LOSS_FN_H

#include <vector>
#include <iostream>
#include <string>
#include <cmath>

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/ParFriends.h"

#include "../utils/configParse.h"
#include "../utils/parDenseGEMM.h"
#include "../DenseMatrix/DenseMatrix.h"
#include "../DenseMatrix/DenseMatrixAlgorithms.h"

using namespace combblas;


template <typename SR, typename NT>
NT CrossEntropyLoss(DenseMatrix<NT>& pred, const std::vector<int>* target, int test_idx, bool sum = false);

template <typename SR, typename NT>
std::vector<NT> LossMetrics(DenseMatrix<NT>& pred, const std::vector<int>* target, int test_idx, bool sum = false);

template <typename SR, typename NT>
DenseMatrix<NT> DerivativeCrossEntropyLoss(DenseMatrix<NT>& pred, const std::vector<int>* target, bool sum_reduction=false);

#endif 