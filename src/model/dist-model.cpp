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

using namespace std;
using namespace combblas;

typedef SpDCCols <int64_t, double> DCCols;
typedef SpParMat <int64_t, double, DCCols> MPI_DCCols;
typedef SpParMat<int64_t, double, SpDCCols<int64_t, double>> SPMAT_DOUBLE;
typedef DenseParMat<int64_t, double> DPMAT_DOUBLE;
typedef FullyDistVec <int64_t, double> DPVEC_DOUBLE;

typedef PlusTimesSRing<double, double> PTFF;

DistModel::DistModel(ConfigProperties &config, int in_dim){
    input_dim = in_dim;
    output_dim = config.model_properties.classes;
    dropout = config.model_properties.dropout_rate;
    vector<int> lay_dim = config.model_properties.hidden_dims;
    number_of_hid_layers = lay_dim.size();
    bool withBias = config.model_properties.with_bias;

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
    // w = DPVEC_DOUBLE(w_std, dvh->getcommgrid());

    if (number_of_hid_layers > 0){
        auto in_conv = DistConv(input_dim, lay_dim[0], withBias);
        layers.push_back(in_conv);
        for (int i = 1; i < number_of_hid_layers; i++){
            auto conv = DistConv(lay_dim[i-1], lay_dim[i], withBias);
            layers.push_back(conv);
            // layers.push_back(new HGNN_conv(lay_dim[i-1], lay_dim[i], withBias));
        }
        auto out_conv = DistConv(lay_dim[number_of_hid_layers-1], output_dim, withBias);
        layers.push_back(out_conv);
    } else {
        // no hidden layers
        auto out_conv = DistConv(input_dim, output_dim, withBias);
        layers.push_back(out_conv);
    }
};

DenseMatrix DistModel::forward(const DenseMatrix &input){
    // dvh times w
    this->G_1 = PSpSCALE<PTFF, int64_t, double, SpDCCols<int64_t, double>>(this->dvh, this->w);
    // G_1.ParallelWriteMM("../data/m_g_ms_gs/bla.mtx", true);

    this->G_1 = PSpGEMM<PTFF, int64_t, double, double, SpDCCols < int64_t, double >, SpDCCols <int64_t, double >>(this->G_1, this->invde_ht_dvh);

    // this->layers[0].G_2 = PSpGEMM<PTFF, int64_t, double, double, SpDCCols < int64_t, double >, SpDCCols <int64_t, double >>(this->G_1, input);

    // DPMAT_DOUBLE x = this->layers[0].forward(input);

    // this->layers[0].G_2 = this->invde_ht_dvh->PSpGEMM(x, false);
}
// torch::Tensor ModelW::forward(const torch::Tensor &input){
//     torch::Tensor ident = torch::eye(dvh->size(1));
//     torch::Tensor w_tmp = torch::diagonal_scatter(ident, w);
//     torch::Tensor x = layers[0]->forward(input);
//     x = invde_ht_dvh->mm(x);
//     x = w_tmp.mm(x);
//     x = dvh->mm(x);
//     x = torch::relu(x);
//     x = torch::dropout(x, this->dropout, true);
//     for (int i = 1; i < number_of_hid_layers; i++){
//         x = layers[i]->forward(x);
//         x = invde_ht_dvh->mm(x);
//         x = w_tmp.mm(x);
//         x = dvh->mm(x);
//         x = torch::relu(x);
//         x = torch::dropout(x, this->dropout, true);
//     }
//     x = layers[layers.size()-1]->forward(x);
//     x = invde_ht_dvh->mm(x);
//     x = w_tmp.mm(x);
//     x = dvh->mm(x);
//     return x;
//     // return torch::nn::functional::softmax(x, torch::nn::functional::SoftmaxFuncOptions(1));
// }

DistConv::DistConv(int in_dim, int out_dim, bool withBias=false){
    //TODO: correct initialization

    shared_ptr<CommGrid> fullWorld;
    fullWorld.reset( new CommGrid(MPI_COMM_WORLD, out_dim, in_dim));

    this->weights = DPMAT_DOUBLE(1.0, fullWorld, out_dim, in_dim);

    if (withBias){
        this->bias = DPVEC_DOUBLE(out_dim, 0.0);
    } 
    // reset_parameters();
}


DPMAT_DOUBLE DistConv::forward(DPMAT_DOUBLE &input){
    // DPMAT_DOUBLE tmp = input * this->weights;
}