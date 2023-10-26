#include "model.h"
#include <torch/torch.h>

#include <vector>
#include <iostream>

Model::Model(int in_dim, std::vector<int> &lay_dim, int out_dim, double dropout_value, bool withBias=false){
    input_dim = in_dim;
    output_dim = out_dim;
    dropout = dropout_value;

    if (lay_dim.size() > 0){
        layers.push_back(new HGNN_conv(input_dim, lay_dim[0], withBias));
        for (int i = 1; i < lay_dim.size(); i++){
            layers.push_back(new HGNN_conv(lay_dim[i-1], lay_dim[i], withBias));
        }

        layers.push_back(new HGNN_conv(lay_dim[lay_dim.size()-1], output_dim, withBias));
        
        for (int i = 0; i < layers.size(); i++){
            std::cout << layers[i] << std::endl;
        }
        std::cout << "size of vector: " << layers.size()  << std::endl;
    }
}



HGNN_conv::HGNN_conv(int in_dim, int out_dim, bool withBias=false){
    weights = torch::rand({in_dim, out_dim});
    //torch::Tensor tmp = torch::rand({in_dim, out_dim});
    // weights = &tmp;
    if (withBias){
        //torch::Tensor bias_tmp = torch::rand({out_dim});
        // bias = &bias_tmp;
        bias = torch::rand({out_dim});

    }

    std::cout << "weights " << weights << std::endl;
    std::cout << "weights address: " << &weights << std::endl;

}




