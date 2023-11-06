#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>
#include <vector>
#include <iostream>


class HGNN_conv : public torch::nn::Module
{
    private:
        torch::nn::Linear linear_layer = nullptr;

    public:
        HGNN_conv(int in_dim, int out_dim, bool withBias, bool t);
        torch::Tensor forward(const torch::Tensor &input);

};

class Model : public torch::nn::Module
{
    private:
        int input_dim;
        std::vector<int>* layer_dim;
        int output_dim;
        int number_of_hid_layers;
        std::vector<std::shared_ptr<HGNN_conv>> layers;
        double dropout;
        const torch::Tensor *leftSide;

    public:
        Model(int in_dim, std::vector<int> lay_dim, int out_dim, double dropout, const torch::Tensor *leftSide, bool withBias);

        // forward function of the Model, it takes the features X (called input) and the constant leftSide of the expression 10 of the paper 
        // Hypergraph Neural Networks (called leftSide)
        torch::Tensor forward(const torch::Tensor &input);


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