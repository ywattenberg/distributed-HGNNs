#include "model.h"
#include <torch/torch.h>

#include <vector>
#include <iostream>
#include <string>
#include <cmath>

Model::Model(int in_dim, std::vector<int> lay_dim, int out_dim, double dropout_value, const torch::Tensor *leftSide, bool withBias=false){
    input_dim = in_dim;
    output_dim = out_dim;
    dropout = dropout_value;
    number_of_hid_layers = lay_dim.size();
    this->leftSide = leftSide;

    if (number_of_hid_layers > 0){
        auto in_conv = std::make_shared<HGNN_conv>(input_dim, lay_dim[0], withBias);
        register_module("HG_conv_in", in_conv);
        layers.push_back(in_conv);
        for (int i = 1; i < number_of_hid_layers; i++){
            auto conv = std::make_shared<HGNN_conv>(lay_dim[i-1], lay_dim[i], withBias);
            std::string module_name = "HG_conv_" + std::to_string(i);
            register_module(module_name, conv);
            layers.push_back(conv);
            // layers.push_back(new HGNN_conv(lay_dim[i-1], lay_dim[i], withBias));
        }
        auto out_conv = std::make_shared<HGNN_conv>(lay_dim[number_of_hid_layers-1], output_dim, withBias);
        register_module("HG_conv_out", out_conv);
        layers.push_back(out_conv);
    } else {
        // no hidden layers
        auto out_conv = std::make_shared<HGNN_conv>(input_dim, output_dim, withBias);
        register_module("HG_conv_out", out_conv);
        layers.push_back(out_conv);
    }
}


torch::Tensor Model::forward(const torch::Tensor &input){
    torch::Tensor x = layers[0]->forward(input);
    x = leftSide->mm(x);
    x = torch::relu(x);
    x = torch::dropout(x, this->dropout, true);
    for (int i = 1; i < number_of_hid_layers; i++){
        x = layers[i]->forward(x);
        x = leftSide->mm(x);
        x = torch::relu(x);
        x = torch::dropout(x, this->dropout, true);
    }
    x = layers[layers.size()-1]->forward(x);
    return torch::nn::functional::softmax(x, torch::nn::functional::SoftmaxFuncOptions(1));
}



HGNN_conv::HGNN_conv(int in_dim, int out_dim, bool withBias=false, bool t=false){
    
    weights = register_parameter("weights", torch::empty({in_dim, out_dim}));
    if (withBias){
        bias = register_parameter("bias", torch::empty({out_dim}));
    } else {
        bias = register_parameter("bias", torch::Tensor());
    }
    reset_parameters();
    //linear_layer = register_module("linear", torch::nn::Linear(in_dim, out_dim));
}

void HGNN_conv::reset_parameters(){
    double stdv = 1.0/std::sqrt(weights.size(1));
    weights.data().uniform_(-stdv, stdv);
    if (bias.defined()){
        bias.data().uniform_(-stdv, stdv);
    }
}

torch::Tensor HGNN_conv::forward(const torch::Tensor &input){
    // torch::Tensor x = linear_layer(input);
    torch::Tensor x = input.mm(weights);
    if (bias.defined()){
        x += bias;
    }

    return x;
}