#include "model.h"
#include <torch/torch.h>

#include <vector>
#include <iostream>

Model::Model(int in_dim, std::vector<int> &lay_dim, int out_dim, double dropout_value, bool withBias=false){
    input_dim = in_dim;
    output_dim = out_dim;
    dropout = dropout_value;
    number_of_hid_layers = lay_dim.size();

    if (number_of_hid_layers > 0){
        layers.push_back(new HGNN_conv(input_dim, lay_dim[0], withBias));
        for (int i = 1; i < number_of_hid_layers; i++){
            layers.push_back(new HGNN_conv(lay_dim[i-1], lay_dim[i], withBias));
        }
        layers.push_back(new HGNN_conv(lay_dim[number_of_hid_layers-1], output_dim, withBias));
    } else {
        // no hidden layers
        layers.push_back(new HGNN_conv(input_dim, output_dim, withBias));
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



HGNN_conv::HGNN_conv(int in_dim, int out_dim, bool withBias=false){
    weights = torch::rand({in_dim, out_dim});
    if (withBias){
        bias = torch::rand({out_dim});
    }
}

torch::Tensor HGNN_conv::forward(torch::Tensor &input, torch::Tensor &leftSide){
    torch::Tensor x = leftSide.mm(input);
    x = x.mm(weights);
    return x;
}




