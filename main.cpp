#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "model/model.h"


int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << &tensor << std::endl;
  std::cout << tensor << std::endl;

  std::vector<int> layers;
  layers.push_back(4);
  layers.push_back(19);

  std::cout << layers[0] << std::endl;

  Model *m = new Model(10, layers, 5, 0.2, false);
   
}