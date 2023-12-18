#ifndef CUSTOM_LOSS_FN_H
#define CUSTOM_LOSS_FN_H

#include <vector>
#include <iostream>
#include <string>
#include <cmath>

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/ParFriends.h"

#include "../utils/DenseMatrix.h"
#include "../utils/configParse.h"
#include "../utils/parDenseGEMM.h"
#include "../utils/DenseMatrix.h"

using namespace combblas;


template <typename SR, typename NT>
NT CrossEntropyLoss(DenseMatrix<NT>& pred, const std::vector<int>* target, bool sum = false);

template <typename SR, typename NT>
DenseMatrix<NT> DerivativeCrossEntropyLoss(DenseMatrix<NT>& pred, const std::vector<int>* target, bool sum_reduction=false);

#endif 