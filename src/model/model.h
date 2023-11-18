#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>
#include <vector>
#include <iostream>


class HGNN_conv : public torch::nn::Module
{
    private:
        torch::Tensor weights;
        torch::Tensor bias;

    public:
        HGNN_conv(int in_dim, int out_dim, bool withBias, bool t);
        torch::Tensor forward(const torch::Tensor &input);
        void reset_parameters();

};

class HGNN_fc : torch::nn::Module
{
    private:
        torch::Tensor weights;
    
    public:
        HGNN_fc(int in_dim, int out_dim);

        torch::Tensor forward(torch::Tensor* input);

};

class BaseModel : public torch::nn::Module {
    private:
        int input_dim;
        std::vector<int>* layer_dim;
        int output_dim;
        int number_of_hid_layers;
        std::vector<std::shared_ptr<HGNN_conv>> layers;
        double dropout;
        const torch::Tensor *leftSide;

    public:
        virtual torch::Tensor forward(const torch::Tensor &input) {
            std::cout << "BaseModel forward, don't use this!!!!!" << std::endl;
            return input;
        }
};

class Model : public BaseModel
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
        torch::Tensor forward(const torch::Tensor &input) override;


};


class ModelW : public BaseModel {
    private:
        int input_dim;
        std::vector<int>* layer_dim;
        int output_dim;
        int number_of_hid_layers;
        std::vector<std::shared_ptr<HGNN_conv>> layers;
        double dropout;
        const torch::Tensor *dvh;
        const torch::Tensor *invde_ht_dvh;
        torch::Tensor w;

    public:
        ModelW(int in_dim, std::vector<int> lay_dim, int out_dim, double dropout, const torch::Tensor *dvh, const torch::Tensor *invde_ht_dvh, bool withBias);

        // forward function of the Model, it takes the features X (called input) and the constant leftSide of the expression 10 of the paper 
        // Hypergraph Neural Networks (called leftSide)
        torch::Tensor forward(const torch::Tensor &input) override;
};


#endif