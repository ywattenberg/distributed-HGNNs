#ifndef DIST_MODEL_H
#define DIST_MODEL_H

#include <vector>
#include <iostream>

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/SpParMat3D.h"
#include "CombBLAS/DenseParMat.h"
#include "CombBLAS/FullyDistVec.h"
#include "CombBLAS/ParFriends.h"

#include "../utils/DenseMatrix.h"
#include "../utils/configParse.h"

using namespace std;
using namespace combblas;

typedef SpDCCols <int64_t, double> DCCols;
typedef SpParMat <int64_t, double, DCCols> MPI_DCCols;
typedef SpParMat<int64_t, double, SpDCCols<int64_t, double>> SPMAT_DOUBLE;
typedef DenseMatrix<double> DENSE_DOUBLE;
typedef FullyDistVec <int64_t, double> DPVEC_DOUBLE;

class DistConv
{
    private:
        DENSE_DOUBLE weights;
        DPVEC_DOUBLE bias;

    public:
        DENSE_DOUBLE* X; // This will hold the input X
        DENSE_DOUBLE Xt; // This will hold the result of X*\theta (or G_2) 
        DENSE_DOUBLE XtB; // This will hold the result of X*\theta + b (also G_2) if bias is used
        DENSE_DOUBLE G_3; // This will hold the result of G_2 * LWR (or G_3)
        DENSE_DOUBLE G_4; // This will hold the result of ReLU(G_3) (or G_4)

        DistConv(int in_dim, int out_dim, bool withBias);
        ~DistConv();
        DENSE_DOUBLE forward(DENSE_DOUBLE &input);
};

class DistModel
{
    private:
        int input_dim;
        std::vector<int>* layer_dim;
        int output_dim;
        int number_of_hid_layers;
        double dropout;
        bool withBias;
        std::vector<DistConv> layers;

        SPMAT_DOUBLE dvh;
        SPMAT_DOUBLE invde_ht_dvh;
        vector<double> w;
        SPMAT_DOUBLE LWR;
        SPMAT_DOUBLE LR;


    public:
        DistModel(ConfigProperties &config, int in_dim);
        ~DistModel();
        // forward function of the Model, it takes the features X (called input) and the constant leftSide of the expression 10 of the paper 
        // Hypergraph Neural Networks (called leftSide)
        DENSE_DOUBLE forward(const DENSE_DOUBLE &input);
        


};

// TODO: Parallelize this function
template <typename NT>
std::vector<NT>* CrossEntropyLoss(const std::vector<NT>* pred, const std::vector<NT>* target);

// //TODO: Parallelize this function
// template <typename NT>
// std::vector<NT>* CrossEntropyLossDerivative(const std::vector<NT>* pred, const std::vector<NT>* target)
// #endif