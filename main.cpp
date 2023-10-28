#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "model/model.h"


int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::vector<float> customData = {0.0, 0.0, 0.0, 4.0, 0.0, 0.0};
  std::vector<int64_t> shape = {3,2};
  tensor = tensor.to_sparse();

  at::Tensor customTensor = torch::from_blob(customData.data(), shape);
  customTensor = customTensor.to_sparse();

  at::Tensor res2 = tensor.mm(customTensor);
  at::Tensor res = torch::mm(tensor, customTensor);

  std::cout << "tensor:\n" << tensor << std::endl;
  std::cout << "sparse:\n" << customTensor << std::endl;
  auto indices = customTensor.indices();
  std::cout << "indices:\n" << indices << std::endl;

  std::cout << "Res 2:\n" << res2 << std::endl;
  std::cout << "Res:\n" << res << std::endl;


  std::vector<int> layers;
  layers.push_back(5);
  layers.push_back(5);

  std::cout << layers[0] << std::endl;

  torch::Tensor input = torch::rand({5,5});
  Model* m = new Model(5, layers, 5, 0.0, false);

  std::cout << "input: " << input << std::endl;

  at::Tensor identityMatrix = torch::eye(5);
  std::cout << "output: " << m->forward(input, identityMatrix, true) << std::endl;
  
  std::cout << "input: " << input << std::endl;


   
}


