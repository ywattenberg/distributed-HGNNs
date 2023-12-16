#include "dist-model.h"
// #include <torch/torch.h>

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
#include "../utils/LossFn.h"
#include "../utils/DerivativeFunctions.h"

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
    this->learning_rate = config.trainer_properties.learning_rate;

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

    // Calculate the product L*R for the backwardpass of W 

    this->LR = PSpGEMM<PTFF, int64_t, double, double, DCCols, DCCols>(this->dvh, this->invde_ht_dvh);
    std::cout << "Reached init of layers" << std::endl;
    //TODO: use correct dimension
    vector<double> w(in_dim, 1.0);
    this->layers = vector<DistConv*>();
    this->layers.reserve(number_of_hid_layers);
    if (number_of_hid_layers > 0){
        auto in_conv = new DistConv(fullWorld, input_dim, lay_dim[0], this->withBias);
        this->layers.push_back(in_conv);
        std::cout << "Reached here" << std::endl;
        for (int i = 1; i < number_of_hid_layers; i++){
            std::cout << "Reached begin loop at: " << i << std::endl;
            auto conv = new DistConv(fullWorld, lay_dim[i-1], lay_dim[i], this->withBias);
            std::cout << "created layer" << std::endl;
            this->layers.push_back(conv);
            std::cout << "Reached loop at: " << i << std::endl;
            // layers.push_back(new HGNN_conv(lay_dim[i-1], lay_dim[i], this->withBias));
        }
        std::cout << "Reached end loop" << std::endl;
        auto out_conv = new DistConv(fullWorld, lay_dim[number_of_hid_layers-1], output_dim, this->withBias);
        std::cout << "created last layer" << std::endl;
        this->layers.push_back(out_conv);
        std::cout << "Finished loop" << std::endl;
    } else {
        // no hidden layers
        auto out_conv = new DistConv(fullWorld, input_dim, output_dim, this->withBias);
        this->layers.push_back(out_conv);
    }
    std::cout << "Finished init" << std::endl;
};

DistModel::~DistModel(){};


void DistModel::comp_layer(DENSE_DOUBLE* X, DistConv* curr, bool last_layer=false){
    // Compute Xt (X * theta or G_2) where both are dense matrices
    curr->XtB = DenseDenseMult<PTFF, double>(*X, curr->weights);
    if (this->withBias){
        // Compute XTb (X * Theta + b or G_2) where both are dense matrices
        curr->XtB.addBiasLocally(&curr->bias);
    }
    // Compute G_3 (LWR * XTb) with bias and (LWR * XT) without, where LWR is a sparse matrix and XTb/XT are dense matrices
    curr->G_3 = SpDenseMult<PTFF, int64_t, double, DCCols>(this->LWR, curr->XtB);
    // Compute X (ReLU(G_3) or G_4) if not last layer
    curr->G_4 = last_layer ? DENSE_DOUBLE() : DenseReLU<PTFF, double>(curr->G_3);
}

DENSE_DOUBLE* DistModel::forward(DENSE_DOUBLE* input){
    // First compute LWR or G_1 (will be the same for all layers)
    this->LWR = PSpSCALE<PTFF, int64_t, double, DCCols>(this->dvh, this->w);
    this->LWR = PSpGEMM<PTFF, int64_t, double, double, DCCols, DCCols>(this->LWR, this->invde_ht_dvh);
    DENSE_DOUBLE* X = input;
    // All other calculations are have to be done for each layer
    for(int i = 0; i < this->layers.size()-1; i++){
        DistConv* curr = this->layers[i];
        curr->X = X;
        // Compute each layer
        comp_layer(X, curr);
        // Set X to G_4 for next iteration
        X = &(curr->G_4);
    }
    // Last layer is different as we do not use ReLU
    X = &(this->layers[this->layers.size()-1]->G_3);
    return X;
}

void DistModel::backward(DENSE_DOUBLE& input, std::vector<int>* labels, double learning_rate){
    // We need to accumulate the gradients of w over all layers reducing using sum
    // As w is a vector we will only need the diagonal of the matrix derivative
    std::vector<double> dw = std::vector<double>(this->w.size(), 0.0);
    
    DENSE_DOUBLE dL_dX = DerivativeCrossEntropyLoss<PTFF, double>(input, labels); 
    DistConv* curr = this->layers[this->layers.size()-1];
    // Last layer is different as we do not use ReLU this means dL_dG3 is just dL_dX
    DENSE_DOUBLE dL_dG1 = DenseDenseMult<PTFF, double>(dL_dX, curr->XtB);
    DENSE_DOUBLE dL_dw  = DenseSpMult<PTFF, int64_t, double, DCCols>(dL_dG1, this->LR);
    DENSE_DOUBLE dL_dG2 = DenseSpMult<PTFF, int64_t, double, DCCols>(dL_dX, this->LWR);
    DENSE_DOUBLE dL_dt  = DenseDenseMult<PTFF, double>(dL_dG2, *curr->X);
    // Set dL_dX for next iteration
    dL_dX  = DenseDenseMult<PTFF, double>(dL_dG2, curr->weights);

    //Now we need to accumulate the gradients of w over
    WDerivativeLocalAdd(dL_dw, &dw);

    // Now we need to step the weights and bias for the last layer
    DenseGradientStep<PTFF, int64_t, double>(&curr->weights, &dL_dt, learning_rate);
    if (this->withBias){
        BiasGradientStep<PTFF, int64_t, double>(&curr->bias, dL_dG2, learning_rate);
    }

    
    for(int i = this->layers.size()-2; i >= 1; i--){
        curr = this->layers[i];
        // Compute the gradients for each layer
        // We are given dL_dX from the previous layer

        DENSE_DOUBLE dX_dG3 = DerivativeDenseReLU<PTFF, double>(curr->G_3);
        DENSE_DOUBLE dL_dG3 = DenseDenseMult<PTFF, double>(dL_dX, dX_dG3);
                     dL_dG1 = DenseDenseMult<PTFF, double>(dL_dG3, curr->XtB);
                     dL_dw  = DenseSpMult<PTFF, int64_t, double, DCCols>(dL_dG1, this->LR);
                     dL_dG2 = DenseSpMult<PTFF, int64_t, double, DCCols>(dL_dG3, this->LWR);
                     dL_dt  = DenseDenseMult<PTFF, double>(dL_dG2, *curr->X);
        // Set dL_dX for next iteration
        dL_dX = DenseDenseMult<PTFF, double>(dL_dG2, curr->weights);
        // Derivate of loss with respect to bias B is just dL_dG2 

        // Update weights and bias
        DenseGradientStep<PTFF, int64_t, double>(&curr->weights, &dL_dt, learning_rate);
        if (this->withBias){
            VecGradientStep<PTFF, int64_t, double>(&curr->bias, &dL_dG2, learning_rate);
        }
    }
}

DistConv::DistConv(){
    this->weights = DENSE_DOUBLE();
    this->bias = vector<double>();
    this->X = new DENSE_DOUBLE();
    this->XtB = DENSE_DOUBLE();
    this->G_3 = DENSE_DOUBLE();
    this->G_4 = DENSE_DOUBLE();
}

//TODO: write implementation of weight initialization
DistConv::DistConv(shared_ptr<CommGrid> fullWorld, int in_dim, int out_dim, bool withBias=false){
    int gridRows = fullWorld->GetGridRows();
    int gridCols = fullWorld->GetGridCols();
    int rankInRow =  fullWorld->GetRankInProcRow();
    int rankInCol = fullWorld->GetRankInProcCol();

    
    int local_rows = in_dim / gridRows;
    int local_cols = out_dim / gridCols;
    if (rankInCol == gridRows - 1){
        local_rows += in_dim % gridRows;
    }

    if (rankInRow == gridCols -1){
        local_cols += out_dim % gridCols;
    }
    
    vector<double>* weight_vec = new vector<double>(local_rows * local_cols, 0.0);
    //TODO: Parallelize 
    double stdv = 1.0 / std::sqrt(in_dim);
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-stdv, stdv);
    for(int i = 0; i < local_rows * local_cols; i++){
        weight_vec->at(i) = dis(gen);
    }

    this->weights = *(new DENSE_DOUBLE(local_rows, local_cols, weight_vec, fullWorld));

    if (withBias){
        this->bias = vector<double>(out_dim, 1.0);
    } 
}