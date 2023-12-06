# ifndef HGNN_H
# define HGNN_H

#include <vector>
#include <utility>
// #include <omp.h>
#include <iostream>

#include "distributed_sddmm/sparse_kernels.h"
#include "distributed_sddmm/SpmatLocal.hpp"
#include "distributed_sddmm/common.h"


/*
For reference:

https://github.com/PASSIONLab/distributed_sddmm/blob/master/gat.hpp
https://github.com/PASSIONLab/distributed_sddmm/blob/master/benchmark_dist.cpp 
*/

class Module {
    private:
        DenseMatrix weights;

     public:
        void BaseModel(){

        }

        //TODO: inplace or return SpMat/DenseMatrix?
        void forward(){

        }

        void backward(){

        }
}

class Dropout: Module {

}

class Linear: Module {
    private:
        DenseMatrix weights;

    public:
        Linear(int in_dim, int out_dim){

        }

        void forward(DenseMatrix input){

        }

}

class HGNN_Conv: Module {
    private:
        DenseMatrix weights;

    public:
        HGNN_Conv(int in_dim, int out_dim, bool withBias, bool t){

        }

        void forward(){

        }

        void backward(){

        }

}

class BaseModel: Module {
    private:
        int input_dim;
        std::vector<int> layer_dim;
        int output_dim;
        int number_of_hid_layers;
        std::vector<std::shared_ptr<HGNN_Conv>> layers; // is shared_ptr necessary?
        double dropout;
        const std::vector<SpParMat<..., >> *leftSide;   // TODO: change this to sparsemat type
        Distributed_Sparse* d_ops;                      // get this from repo
        vector<DenseMatrix> buffers;

    public:
        void BaseModel(){

        }

        void forward(){

        }

        void backward(){

        }
}   

class Model: public BaseModel {
    private:
        int input_dim;
        std::vector<int> layer_dim;
        int output_dim;
        int number_of_hid_layers;
        std::vector<std::shared_ptr<HGNN_Conv>> layers; 
        double dropout;
        const SpParMat *leftSide; 
        Distributed_Sparse* d_ops; 

    public:
        BaseModel(int in_dim, std::vector<int> lay_dim, int out_dim, double dropout, const SpParMat *leftSide; *leftSide, bool withBias);){

        }

        void forward(){

        }

        void backward(){

        }


}

class ModelW: public BaseModel {

    private:
        int input_dim;
        std::vector<int> layer_dim;
        int output_dim;
        int number_of_hid_layers;
        std::vector<std::shared_ptr<HGNN_Conv>> layers; 
        double dropout;
        const SpParMat *leftSide;
        Distributed_Sparse* d_ops; 

    public:
        ModelW(int in_dim, std::vector<int> lay_dim, int out_dim, double dropout, const SpParMat *leftSide, bool withBias){
            this->input_dim = in_dim;
        }

        void forward(){

        }

        void backward(){

        }

}


#endif


