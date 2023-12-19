#include "dist-model_no_w.h"
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

DistModelW::DistModelW(ConfigProperties &config, int in_dim, std::shared_ptr<CommGrid> grid, int dim_w){
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

    this->fullWorld = grid;
    // shared_ptr<CommGrid> fullWorld;
	// fullWorld.reset(new CommGrid(MPI_COMM_WORLD, std::sqrt(size), std::sqrt(size)));

    // read data
    SPMAT_DOUBLE dvh(fullWorld);
    SPMAT_DOUBLE invde_ht_dvh(fullWorld);
    dvh.ParallelReadMM(config.data_properties.dvh_path, true, maximum<double>());
    invde_ht_dvh.ParallelReadMM(config.data_properties.invde_ht_dvh_path, true, maximum<double>());

    this->LWR = PSpGEMM<PTFF, int64_t, double, double, DCCols, DCCols>(dvh, invde_ht_dvh);
    this->LWR_T = SPMAT_DOUBLE(this->LWR);
    this->LWR_T.Transpose();

    // Calculate the product L*R for the backwardpass of W 

    this->layers = vector<DistConvW*>();
    this->layers.reserve(number_of_hid_layers);
    if (number_of_hid_layers > 0){
        auto in_conv = new DistConvW(fullWorld, input_dim, lay_dim[0], this->withBias);
        this->layers.push_back(in_conv);
        for (int i = 1; i < number_of_hid_layers; i++){
            auto conv = new DistConvW(fullWorld, lay_dim[i-1], lay_dim[i], this->withBias);
            this->layers.push_back(conv);
            // layers.push_back(new HGNN_conv(lay_dim[i-1], lay_dim[i], this->withBias));
        }
        auto out_conv = new DistConvW(fullWorld, lay_dim[number_of_hid_layers-1], output_dim, this->withBias);
        this->layers.push_back(out_conv);
    } else {
        // no hidden layers
        auto out_conv = new DistConvW(fullWorld, input_dim, output_dim, this->withBias);
        this->layers.push_back(out_conv);
    }
    std::cout << "Finished init" << std::endl;
}


void DistModelW::comp_layer(DENSE_DOUBLE& X, DistConvW* curr, bool last_layer){
    // Compute Xt (X * theta or G_2) where both are dense matrices
    MPI_Barrier(MPI_COMM_WORLD);
    int totalRows = X.getnrow();
    int totalCols = X.getncol();

    int totalRowsW = curr->weights.getnrow();
    int totalColsW = curr->weights.getncol();
    
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    
    // DenseMatrix<double> tmptmp = DenseDenseMult<PTFF, double>(*X, curr->weights);
    curr->XtB = DenseDenseMult<PTFF, double>(X, curr->weights);
    // curr->XtB = *(new DenseMatrix<double>(1,1, test, curr->weights.getCommGrid()));
    if (this->withBias){
        curr->XtB.addBiasLocally(&curr->bias);
    }

    // Compute G_3 (LWR * XTb) with bias and (LWR * XT) without, where LWR is a sparse matrix and XTb/XT are dense matrices
    curr->G_3 = SpDenseMult<PTFF, int64_t, double, DCCols>(this->LWR, curr->XtB);

    // Compute X (ReLU(G_3) or G_4) if not last layer
    curr->G_4 = last_layer ? curr->G_3 : DenseReLU<PTFF, double>(curr->G_3);
    return;
}

void DistModelW::clear_layer_partial_results(){
    for(int i = 0; i < this->layers.size(); i++){
        //For each layer free all partial results saved in the layer
        DistConvW* curr = this->layers[i];
        MPI_Barrier(MPI_COMM_WORLD);
        curr->clear_partial_results(i == this->layers.size()-1);
    }

}

DENSE_DOUBLE DistModelW::forward(DENSE_DOUBLE& input){
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    // First compute LWR or G_1 (will be the same for all layers)
    DENSE_DOUBLE X = input;
    // All other calculations are have to be done for each layer
    for(int i = 0; i < this->layers.size(); i++){
        DistConvW* curr = this->layers[i];
        curr->X = X;
        // Compute each layer
        comp_layer(X, curr, i == this->layers.size()-1);
        // Set X to G_4 for next iteration
        X = curr->G_4;
    }
    // Last layer is different as we do not use ReLU
    return this->layers[this->layers.size()-1]->G_3;
}



void DistModelW::backward(DENSE_DOUBLE& input, std::vector<int>* labels, double learning_rate){
    // We need to accumulate the gradients of w over all layers reducing using sum
    // As w is a vector we will only need the diagonal of the matrix derivative
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    DENSE_DOUBLE dL_dX = DerivativeCrossEntropyLoss<PTFF, double>(input, labels); 
    DistConvW* curr = this->layers[this->layers.size()-1];
    // Last layer is different as we do not use ReLU this means dL_dG3 is just dL_dX
    DENSE_DOUBLE dL_dG2 = SpDenseMult<PTFF, int64_t, double, DCCols>(this->LWR_T, dL_dX);
    DENSE_DOUBLE dL_dt  = DenseTransDenseMult<PTFF, double>(curr->X, dL_dG2);    

    // Set dL_dX for next iteration
    dL_dX.clear();
    dL_dX = DenseDenseTransMult<PTFF, double>(dL_dG2, curr->weights);
    // Now we need to upadte the weights and bias for the last layer
    DenseGradientStep<PTFF, int64_t, double>(curr->weights, dL_dt, learning_rate);
    if (this->withBias){
        BiasGradientStep<PTFF, int64_t, double>(&curr->bias, dL_dG2, learning_rate);
    }
    // Clear all created DenseMatrices except dL_dX
    dL_dG2.clear();
    dL_dt.clear();
    for(int i = this->layers.size()-2; i >= 0; i--){
        curr = this->layers[i];
        // Compute the gradients for each layer
        // We are given dL_dX from the previous layer
        DENSE_DOUBLE dX_dG3 = DerivativeDenseReLU<PTFF, double>(curr->G_3);
        DENSE_DOUBLE dL_dG3 = DenseElementWiseMult<PTFF, double>(dL_dX, dX_dG3);
                     dL_dG2 = SpDenseMult<PTFF, int64_t, double, DCCols>(this->LWR_T, dL_dX);
                     dL_dt  = DenseTransDenseMult<PTFF, double>(curr->X, dL_dG2); 
        
        // Update weights and bias
        DenseGradientStep<PTFF, int64_t, double>(curr->weights, dL_dt, learning_rate);
        if (this->withBias){
            BiasGradientStep<PTFF, int64_t, double>(&curr->bias, dL_dG2, learning_rate);
        }
        // Set dL_dX for next iteration
        dL_dX.clear();
        dL_dX = DenseDenseTransMult<PTFF, double>(dL_dG2, curr->weights);
        // Clear all created DenseMatrices except dL_dX
        dL_dG2.clear();
        dL_dt.clear();
        dL_dG3.clear();
        dX_dG3.clear();

    }
    dL_dX.clear();
    this->clear_layer_partial_results();

}


DistConvW::DistConvW(){
    this->weights = DENSE_DOUBLE();
    this->bias = vector<double>();
    this->X = DENSE_DOUBLE();
    this->XtB = DENSE_DOUBLE();
    this->G_3 = DENSE_DOUBLE();
    this->G_4 = DENSE_DOUBLE();
}

void DistConvW::clear_partial_results(bool last_layer){
    if(this->XtB.getValues() != nullptr){
        this->XtB.clear();
    }
    if(this->G_3.getValues() != nullptr){
        this->G_3.clear();
    }
    if(!last_layer && this->G_4.getValues() != nullptr){
        this->G_4.clear();
    }
}

DistConvW::DistConvW(shared_ptr<CommGrid> fullWorld, int in_dim, int out_dim, bool withBias=false){
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
    double stdv = 1.0 / (std::sqrt(in_dim));

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-stdv, stdv);
        
    for(int i = 0; i < local_rows * local_cols; i++){
        weight_vec->at(i) = dis(gen);
    }

    this->weights = DENSE_DOUBLE(local_rows, local_cols, weight_vec, fullWorld);

    if (withBias){
        this->bias = vector<double>(out_dim, 1.0);
    } 
}