#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "model/model.h"

//cmake -Bbuild -DCMAKE_BUILD_TYPE=Release
int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  tensor.print();

  std::vector<int> layers;
  layers.push_back(10);
  layers.push_back(19);

  std::cout << layers[0] << std::endl;

  // create a model with input dim = 10, hidden dims = layers, out_dim = 5
  // dropout = 0.2, withBias = False
  Model *m = new Model(4, layers, 5, 0.2, false);

  return 0;
}

/*
  ref: 
  https://github.com/krshrimali/Digit-Recognition-MNIST-SVHN-PyTorch-CPP/blob/master/training.cpp

  TODOs: 
    -load data
    -compute statistics
*/
void train(){

  // init model 
  int in_dim = 10;
  int out_dim = 5;
  float dropout_rate = 0.2;
  bool withBias = false;

  int dims[] = {10, 15, 10};
  std::vector<int> hidden_dims;
  for (int dim: dims){
    hidden_dims.push_back(10);
  } 

  auto model = std::make_shared<Model>(in_dim, hidden_dims, out_dim, dropout_rate, withBias);

  // init training params
  torch::optim::SGD optimizer(model->parameters(), 0.01); // Learning Rate 0.01
  size_t n_epochs = 50;
  size_t n_batches = 100;
  auto loss_fn = static_cast<torch::Tensor(*)(const torch::Tensor&)>(torch::log_softmax);

  // load data TODO: replace this 
  auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
			std::move(torch::data::datasets::MNIST("../data").map(torch::data::transforms::Normalize<>(0.13707, 0.3081)).map(
				torch::data::transforms::Stack<>())), 64);


  // training loop
  for(size_t epoch=1; epoch <= n_epochs; ++epoch) {
		size_t batch_index = 0;
		for (auto& batch: *data_loader) {
			optimizer.zero_grad();
			torch::Tensor prediction = model->forward(batch.data, (torch::Tensor) torch::rand({1, 1}), false); // TODO: replace
			torch::Tensor loss = loss_fn(prediction, batch.target);
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

  //compute stats

}