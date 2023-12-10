#ifndef DIST_MODEL_H
#define DIST_MODEL_H

#include <torch/torch.h>
#include <vector>
#include <iostream>

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/SpParMat3D.h"
#include "CombBLAS/DenseParMat.h"
#include "CombBLAS/FullyDistVec.h"
#include "CombBLAS/ParFriends.h"

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
        DENSE_DOUBLE Xt; // This will hold the result of X*\theta (or G_2) 
        DENSE_DOUBLE XtB; // This will hold the result of X*\theta + b (also G_2) if bias is used
        DENSE_DOUBLE G_3; 

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

    public:
        DistModel(ConfigProperties &config, int in_dim);
        ~DistModel();
        // forward function of the Model, it takes the features X (called input) and the constant leftSide of the expression 10 of the paper 
        // Hypergraph Neural Networks (called leftSide)
        DENSE_DOUBLE forward(const DENSE_DOUBLE &input);
        


};
#endif