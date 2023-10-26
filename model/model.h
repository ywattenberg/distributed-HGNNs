#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>
#include <vector>
#include <iostream>


class HGNN_conv : torch::nn::Module
{
    private: 
        torch::Tensor weights;
        torch::Tensor bias;

    public:
        HGNN_conv(int in_dim, int out_dim, bool withBias);

        // TODO: add Graph as parameter
        torch::Tensor forward(torch::Tensor &input, torch::Tensor &leftSide);

};

class Model : torch::nn::Module
{
    private:
        // TODO: Graph class
        int input_dim;
        std::vector<int>* layer_dim;
        int output_dim;
        int number_of_hid_layers;
        std::vector<HGNN_conv*> layers;
        double dropout;

    public:
        Model(int in_dim, std::vector<int> &lay_dim, int out_dim, double dropout, bool withBias);

        // forward function of the Model, it takes the features X (called input) and the constant leftSide of the expression 10 of the paper 
        // Hypergraph Neural Networks (called leftSide)
        torch::Tensor forward(torch::Tensor &input, torch::Tensor &leftSide, bool train);


};


class HGNN_fc : torch::nn::Module
{
    private:
        torch::Tensor* weights;
    
    public:
        HGNN_fc(int in_dim, int out_dim);

        torch::Tensor forward(torch::Tensor* input);

};


#endif