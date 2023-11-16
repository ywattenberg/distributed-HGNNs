#include "model.h"

#include <torch/torch.h>
#include <omp.h>

#include <vector>
#include <iostream>
#include <chrono>
#include <fstream>

torch::Tensor customMatrixMultiplication(const torch::Tensor &A, const torch::Tensor &B) {
    // Ensure that A and B are 2D matrices
    assert(A.dim() == 2 && B.dim() == 2);
    assert(A.size(1) == B.size(0));

    auto A_rows = A.size(0);
    auto A_cols = A.size(1);
    auto B_cols = B.size(1);

    // Resultant matrix
    auto C = torch::zeros({A_rows, B_cols}, torch::kFloat);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            for (int k = 0; k < A_cols; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}


Model::Model(int in_dim, std::vector<int> lay_dim, int out_dim, double dropout_value, const torch::Tensor *leftSide, bool withBias=false){
    input_dim = in_dim;
    output_dim = out_dim;
    dropout = dropout_value;
    number_of_hid_layers = lay_dim.size();
    this->leftSide = leftSide;
    // this->mm_function = [](const torch::Tensor &A, const torch::Tensor &B){
    //     return customMatrixMultiplication(A, B);
    // }
    this->mm_function = [=](const torch::Tensor &A, const torch::Tensor &B){
        return torch::mm(A, B);
    };

    if (number_of_hid_layers > 0){
        auto in_conv = std::make_shared<HGNN_conv>(input_dim, lay_dim[0], withBias);
        register_module("HG_conv_in", in_conv);
        layers.push_back(in_conv);
        for (int i = 1; i < number_of_hid_layers; i++){
            auto conv = std::make_shared<HGNN_conv>(lay_dim[i-1], lay_dim[i], withBias);
            std::string module_name = "HG_conv_" + std::to_string(i);
            register_module(module_name, conv);
            layers.push_back(conv);
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
    std::cout << "actual threads used: " << at::get_num_interop_threads() << std::endl;

    // start timing
    auto start = std::chrono::high_resolution_clock::now();

    // the actual forward computation goes here
    torch::Tensor x = mm_function(*leftSide, input);//leftSide->mm(input);//
    x = torch::relu(layers[0]->forward(x));
    for (int i = 1; i < number_of_hid_layers; i++){
        x = mm_function(*leftSide, x); // leftSide->mm(x);
        x = torch::relu(layers[i]->forward(x));
        x = torch::dropout(x, this->dropout, true);
    }
    x = layers[layers.size()-1]->forward(x);
    x = torch::nn::functional::softmax(x, torch::nn::functional::SoftmaxFuncOptions(1));

    // stop timing and save result
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Time taken per forward pass: " 
              << duration.count() << "ms" << std::endl;
    std::ofstream outFile("../timing_info.txt", std::ios::app);
    outFile << duration.count() << std::endl;
    outFile.close();

    return x;
}


HGNN_conv::HGNN_conv(int in_dim, int out_dim, bool withBias=false, bool t=false){
    linear_layer = register_module("linear", torch::nn::Linear(in_dim, out_dim));
}

torch::Tensor HGNN_conv::forward(const torch::Tensor &input){
    // torch::Tensor x = linear_layer(input);
    return linear_layer(input);
}