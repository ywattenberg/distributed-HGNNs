#ifndef _DERIVATIVE_FUNCTIONS_H
#define _DERIVATIVE_FUNCTIONS_H

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

void WDerivativeLocalAdd(DENSE_DOUBLE& dL_dw, std::vector<double>* out);

void WDerivativeAccumulation(CommGrid* commGrid, std::vector<double>* local_w, std::vector<double>* out);

template<typename SR, typename IT, typename NT>
void DenseGradientStep(DenseMatrix<NT>* parameter, DenseMatrix<NT>* gradient, double lr);


template<typename SR, typename IT, typename NT>
void VecGradientStep(std::vector<double>* parameter, DenseMatrix<NT>* gradient, double lr);



#endif