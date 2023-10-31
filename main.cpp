#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "model/model.h"


/*
  ref: 
  https://github.com/krshrimali/Digit-Recognition-MNIST-SVHN-PyTorch-CPP/blob/master/training.cpp

  TODOs: 
    -custom loss_fn
    -load data
*/
void train(){

  // init model 
  int in_dim = 10;
  int out_dim = 10;
  float dropout_rate = 0.2;
  bool withBias = false;

  int dims[] = {10, 10, 10}; //all dimensions are equal because HGNN_conv forward is not implemented yet
  std::vector<int> hidden_dims;
  for (int dim: dims){
    hidden_dims.push_back(dim);
  } 

  auto model = std::make_shared<Model>(in_dim, hidden_dims, out_dim, dropout_rate, withBias);

  // init training params
  for (auto& module : model->named_children()) {
    std::cout << module.key() << ": " << module.value() << std::endl;
  }
  torch::optim::SGD optimizer(model->parameters(), 0.01);
  size_t n_epochs = 50;
  size_t n_batches = 100;


  // training loop
  model->train();

  for(size_t epoch=1; epoch <= n_epochs; ++epoch) {
		size_t batch_index = 0;
      for(int i = 0 ; i < 42; ++i){
        optimizer.zero_grad();
        //TODO: replace the following lines with data loader
        torch::Tensor dummy_input = torch::rand({in_dim, in_dim});
        torch::Tensor dummy_leftSide = torch::rand({in_dim, in_dim});
        torch::Tensor dummy_target = torch::rand({out_dim, out_dim});
        dummy_input.requires_grad_();
        dummy_leftSide.requires_grad_();
        dummy_target.requires_grad_();
        torch::Tensor prediction = model->forward(dummy_input, dummy_leftSide, false); 
        torch::Tensor loss = torch::cross_entropy_loss(prediction, dummy_target); // TODO: replace with custom loss function
        loss.backward();
        optimizer.step();

			  // Output the loss and checkpoint every n_batches
        if (++batch_index % n_batches == 0) {
          std::cout << "Epoch: " << epoch << " | Batch: " << batch_index 
            << " | Loss: " << loss.item<float>() << std::endl;
          torch::save(model, "net.pt");
        }
		}
	}
}


//cmake -Bbuild -DCMAKE_BUILD_TYPE=Release
int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  tensor.print();

  std::vector<int> layers;
  layers.push_back(10);
  layers.push_back(19);

  std::cout << layers[0] << std::endl;

  // test train function
  std::cout << "start training..." << std::endl;
  train();

  return 0;
}
