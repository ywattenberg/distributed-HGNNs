/*
    TODOs:

    1. use register_module so that the weights and params are accessible during trainign
    2. training loop
    3. custom activation function
*/

#include "model.h"
#include <torch/torch.h>

#include <vector>
#include <iostream>

HGNN_conv::HGNN_conv(int in_dim, int out_dim, bool withBias=false){
    // TODO: get D_v, H, D_e, W 
    weights = torch::rand({in_dim, out_dim});
    if (withBias){
        bias = torch::rand({out_dim});
    }
}

torch::Tensor HGNN_conv::forward(torch::Tensor &input, torch::Tensor &leftSide){
    //TODO: determine leftSide
    torch::Tensor x = leftSide.mm(input);
    x = x.mm(weights);
    return x;
}

Model::Model(int in_dim, std::vector<int> &lay_dim, int out_dim, double dropout_value, bool withBias=false){
    input_dim = in_dim;
    output_dim = out_dim;
    dropout = dropout_value;
    number_of_hid_layers = lay_dim.size();
    torch::nn::Sequential layers;
    
    // register modules and wrap into nn::Sequential container
    if (number_of_hid_layers > 0){
        auto in_conv = std::make_shared<HGNN_conv>(input_dim, lay_dim[0], withBias);
        register_module("HG_conv_in", in_conv);
        layers->push_back(in_conv);

        for (int i = 1; i < number_of_hid_layers; i++){
            auto conv = std::make_shared<HGNN_conv>(lay_dim[i-1], lay_dim[i], withBias);
            std::string module_name = "HG_conv_" + std::to_string(i);
            register_module(module_name, conv);
            layers->push_back(conv);
        }

        auto out_conv = std::make_shared<HGNN_conv>(lay_dim[number_of_hid_layers-1], output_dim, withBias);
        register_module("HG_conv_out", out_conv);
        layers->push_back(out_conv);

    } else {
        // no hidden layers
        auto out_conv = HGNN_conv(input_dim, output_dim, withBias);
        register_module("HG_conv_out", out_conv);
        layers->push_back(out_conv);
    }
}


torch::Tensor Model::forward(torch::Tensor &input, torch::Tensor &leftSide, bool train){
    torch::Tensor x;
    x = torch::relu(layers[0]->forward(input, leftSide));
    for (int i = 1; i < number_of_hid_layers; i++){
        x = torch::relu(layers[i]->forward(x, leftSide));
        x = torch::dropout(x, this->dropout, train);
    }
    x = layers[0]->forward(x, leftSide);
    return x;
}



