#include <gtest/gtest.h>
#include <torch/torch.h>
#include <iostream>

#include "../utils/scores.h"


TEST(ScoreFunctionTests, BasicAccuracyTest) {
  std::vector<double> pred_vec = std::vector<double>();
  for(double i = 0.0; i < 1; i+=0.1){
    pred_vec.push_back(i);
    pred_vec.push_back(1-i);
  }
  torch::Tensor pred = torch::from_blob(pred_vec.data(), {10,2}, torch::kDouble).clone();
  torch::Tensor expected = torch::ones({10});
  std::cout << pred << std::endl;
  std::cout << expected << std::endl;
  double true_accuracy = 0.5;
  double clac_accuracy = accuracy(pred, expected).item<double>();
  EXPECT_DOUBLE_EQ(clac_accuracy, true_accuracy) << "Accuracy score is not correct expected: " << true_accuracy << " pred: " << clac_accuracy;
}