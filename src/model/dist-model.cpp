#include "dist-model.h"
#include <torch/torch.h>

#include <vector>
#include <iostream>
#include <string>
#include <cmath>

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/SpParMat3D.h"
#include "CombBLAS/DenseParMat.h"
#include "CombBLAS/FullyDistVec.h"
#include "CombBLAS/ParFriends.h"

#include "../utils/DenseMatrix.h"
#include "../utils/configParse.h"
#include "../utils/parDenseGEMM.h"
#include "../utils/DenseMatrix.h"

using namespace std;
using namespace combblas;

typedef SpDCCols <int64_t, double> DCCols;
typedef SpParMat <int64_t, double, DCCols> MPI_DCCols;
typedef SpParMat<int64_t, double, SpDCCols<int64_t, double>> SPMAT_DOUBLE;
typedef DenseMatrix<double> DENSE_DOUBLE;

typedef PlusTimesSRing<double, double> PTFF;

DistModel::DistModel(ConfigProperties &config, int in_dim){
    input_dim = in_dim;
    output_dim = config.model_properties.classes;
    dropout = config.model_properties.dropout_rate;
    vector<int> lay_dim = config.model_properties.hidden_dims;
    number_of_hid_layers = lay_dim.size();
    this->withBias = config.model_properties.with_bias;

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    shared_ptr<CommGrid> fullWorld;
	fullWorld.reset(new CommGrid(MPI_COMM_WORLD, std::sqrt(size), std::sqrt(size)));

    // read data
    SPMAT_DOUBLE dvh(fullWorld);
    SPMAT_DOUBLE invde_ht_dvh(fullWorld);
    dvh.ParallelReadMM(config.data_properties.dvh_path, true, maximum<double>());
    invde_ht_dvh.ParallelReadMM(config.data_properties.invde_ht_dvh_path, true, maximum<double>());
    this->dvh = dvh;
    this->invde_ht_dvh = invde_ht_dvh;

    //TODO: use correct dimension
    vector<double> w(in_dim, 1.0);
    this->layers = vector<DistConv>();
    this->layers.reserve(number_of_hid_layers);
    if (number_of_hid_layers > 0){
        auto in_conv = DistConv(input_dim, lay_dim[0], this->withBias);
        this->layers.push_back(in_conv);
        for (int i = 1; i < number_of_hid_layers; i++){
            auto conv = DistConv(lay_dim[i-1], lay_dim[i], this->withBias);
            this->layers.push_back(conv);
            // layers.push_back(new HGNN_conv(lay_dim[i-1], lay_dim[i], this->withBias));
        }
        auto out_conv = DistConv(lay_dim[number_of_hid_layers-1], output_dim, this->withBias);
        this->layers.push_back(out_conv);
    } else {
        // no hidden layers
        auto out_conv = DistConv(input_dim, output_dim, this->withBias);
        this->layers.push_back(out_conv);
    }
};

DistModel::~DistModel(){};

DENSE_DOUBLE* DistModel::forward(const DENSE_DOUBLE* input){
    // First compute LWR or G_1 (will be the same for all layers)
    this->LWR = PSpSCALE<PTFF, int64_t, double, DCCols>(this->dvh, this->w);
    this->LWR = PSpGEMM<PTFF, int64_t, double, double, DCCols, DCCols>(this->LWR, this->invde_ht_dvh);
    DENSE_DOUBLE* X = input;
    // All other calculations are have to be done for each layer
    for(int i = 1; i < this->layers.size(); i++){
        DistConv curr = this->layers[i];
        // Compute Xt (X * theta or G_2) where both are dense matrices
        curr.Xt = PDGEMM(X, curr.weights);
        if (this->withBias){
            // Compute XTb (X * Theta + b or G_2) where both are dense matrices
            curr.XtB = DenseDenseAdd(curr.Xt, curr.bias);
        }
        // Compute G_3 (LWR * XTb) with bias and (LWR * XT) without, where both are dense matrices
        curr.G_3 = fox<PTFF, int64_t, double, DCCols>(this->LWR, this->withBias ? curr.XtB : curr.Xt);
        // Compute X (ReLU(G_3) or G_4) where both are dense matrices
        curr.G_4 = DenseReLU(curr.G_3);
        // Set X to G_4 for next iteration
        X = &curr.G_4;
    }
    return X;
    // this->layers[0].G_2 = PSpGEMM<PTFF, int64_t, double, double, SpDCCols < int64_t, double >, SpDCCols <int64_t, double >>(this->G_1, input);

    // DPMAT_DOUBLE x = this->layers[0].forward(input);

    // this->layers[0].G_2 = this->invde_ht_dvh->PSpGEMM(x, false);
}


DistConv::DistConv(int in_dim, int out_dim, bool withBias=false){
    //TODO: correct initialization

    shared_ptr<CommGrid> fullWorld;
    fullWorld.reset( new CommGrid(MPI_COMM_WORLD, out_dim, in_dim));

    this->weights = DENSE_DOUBLE(1.0, fullWorld, out_dim, in_dim);

    if (withBias){
        this->bias = DPVEC_DOUBLE(out_dim, 0.0);
    } 
    // reset_parameters();
}

DistConv::~DistConv(){
    delete this->weights;
    delete this->bias;
    if (this->Xt != NULL){
        delete this->Xt;
    }
    if (this->G_3 != NULL){
        delete this->G_3;
    }
}


DENSE_DOUBLE DistConv::forward(DENSE_DOUBLE &input){
    // DPMAT_DOUBLE tmp = input * this->weights;
}