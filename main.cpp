#include "utils/fileParse.h"
#include <iostream>
#include "model/model.h"
#include "trainer/trainer.h"

// MODEL PARAMETERS
int DATA_SAMPLES = 10;
int FEATURE_DIMENSIONS = 6145;
std::vector<int> HIDDEN_DIMS = {20,10};
double DROPOUT = 0.2;
bool WITH_BIAS = false;
double LEARNING_RATE = 0.001;
int CLASSES = 2;
int EPOCHS = 1000;
int OUTPUT_STEPSIZE = 100; //interval of epochs to output the loss

using LossFunction = at::Tensor(*)(const at::Tensor&, const at::Tensor&); //Supertype for loss functions

int main(){
  std::string G_path = "../data/G_coo.csv";
  std::string Labels_path = "../data/lbls_m_g_ms_gs.csv";
  std::string Features_path = "../data/fts_m_g_ms_gs.csv";

  std::vector<float> data;
  auto [G_lines, G_cols] = csvToArray(std::move(G_path), data);
  torch::Tensor leftSide = torch::from_blob(data.data(), {G_lines,G_cols});
  std::cout << "G dimensions: " << G_lines << "x" << G_cols << std::endl;
  
  // Convert the leftSide tensor to a sparse tensor
  torch::Tensor index = leftSide.index({at::indexing::Slice(), at::indexing::Slice(0,2)}).transpose(0,1);
  index = index.to(torch::kLong);
  torch::Tensor values = leftSide.index({at::indexing::Slice(), at::indexing::Slice(2,3)}).squeeze();
  // This way of calling sparse_coo_tensor assumes 
  // that we have at least one non-zero value in each row/column 
  std::cout << "index dimensions: " << index.sizes() << std::endl;
  std::cout << "values dimensions: " << values.sizes() << std::endl;
  leftSide = torch::sparse_coo_tensor(index, values);
  std::cout << "leftSide dimensions: " << leftSide.sizes() << std::endl;

  std::vector<float> data2;
  auto [L_lines, L_cols] = csvToArray(std::move(Labels_path), data2);
  torch::Tensor labels = torch::from_blob(data2.data(), {L_lines,L_cols});
  std::cout << "Labels dimensions: " << L_lines << "x" << L_cols << std::endl;
 
  std::vector<float> data3; 
  auto [F_lines, F_cols] = csvToArray(std::move(Features_path), data3);
  torch::Tensor features = torch::from_blob(data3.data(), {F_lines,F_cols});
  std::cout << "Features dimensions: " << F_lines << "x" << F_cols << std::endl;


  // Build Model
  auto model = new Model(FEATURE_DIMENSIONS, HIDDEN_DIMS, CLASSES, DROPOUT, &leftSide, WITH_BIAS);

  // Define the loss function
  LossFunction ce_loss_fn = [](const torch::Tensor& predicted, const torch::Tensor& target) {
        return torch::nn::functional::cross_entropy(predicted, target);
    };
  // Train the model
  train_model(EPOCHS, OUTPUT_STEPSIZE, labels, features, ce_loss_fn, model, LEARNING_RATE);

}
