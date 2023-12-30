#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "trainer.h"

#include "../model/model.h"
#include "../utils/scores.h"
#include "../utils/configParse.h"

#define BILLION 1000000000L
#define MILLION 1000000L

using LossFunction = at::Tensor(*)(const at::Tensor&, const at::Tensor&);


using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

void train_model(const ConfigProperties& config, torch::Tensor &labels, torch::Tensor &input_features, LossFunction loss_fn, BaseModel *model, int run_id, bool timing, std::string timing_file){
    
    double lr = config.trainer_properties.learning_rate;
    int n_epochs = config.trainer_properties.epochs;
    int stepsize_output = config.trainer_properties.output_stepsize;
    long train_set_cutoff = config.data_properties.test_idx;

    int lr_step_size = config.lr_scheduler_properties.step_size;
    double lr_gamma = config.lr_scheduler_properties.gamma;

    torch::Tensor train_labels = labels.index({at::indexing::Slice(0,train_set_cutoff)});
    torch::Tensor test_labels = labels.index({at::indexing::Slice(train_set_cutoff,labels.size(0))});

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(lr).weight_decay(0.0005));

    torch::optim::StepLR lr_scheduler = torch::optim::StepLR(optimizer, lr_step_size, lr_gamma);
     
    auto t1 = high_resolution_clock::now();
    for (int epoch = 0; epoch < n_epochs; epoch++){

        torch::Tensor predictions = model->forward(input_features);
        // std::cout << "predictions shape: " << predictions.sizes() << std::endl;
        // // print random predictions
        // for (int i = 0; i < 10; i++){
        //     int idx = rand() % predictions.size(0);
        //     std::cout << "prediction " << idx << ": " << predictions[idx] << std::endl;
        // }
        
        torch::Tensor train_predictions = predictions.index({at::indexing::Slice(0,train_set_cutoff)});
        torch::Tensor loss = loss_fn(train_predictions, train_labels);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        lr_scheduler.step();

        if (epoch % stepsize_output == 0){
            {

                auto t2 = high_resolution_clock::now();
                /* Getting number of milliseconds as an integer. */
                auto ms_int = duration_cast<milliseconds>(t2 - t1);
                /* Getting number of milliseconds as a double. */
                duration<double, std::milli> ms_double = t2 - t1;

                torch::NoGradGuard no_grad;
                torch::Tensor test_predictions = predictions.index({at::indexing::Slice(train_set_cutoff,labels.size(0))});
                torch::Tensor test_loss = loss_fn(test_predictions, test_labels);
                torch::Tensor acc = accuracy(test_predictions, test_labels);
                torch::Tensor f1 = f1_score(test_predictions, test_labels, 40);

                std::cout << "Epoch [" << epoch << "/" << n_epochs << "], " << ms_int.count() << "ms, ";
                std::cout << "Train Loss: " << loss.item<double>() << ", ";
                std::cout << "Test Loss: " << test_loss.item<double>() << ", ";
                std::cout << "Test Accuracy: " << acc.item<double>() << ", ";
                std::cout << "Test F1: " << f1.item<double>() << std::endl;

                if (timing){
                    std::ofstream outfile;
                    outfile.open(timing_file, std::ios_base::app);
                    // append time to csv file
                    outfile << run_id << "," << epoch << "," << ms_int.count() << "," << loss.item<double>() << "," << test_loss.item<double>() << "," << acc.item<double>() << "," << f1.item<double>() << "," << config.model_properties.distributed << "," << config.model_properties.hidden_dims << "," << config.model_properties.learnable_w << "," << config.model_properties.with_bias << "," << config.model_properties.dropout_rate << "," << config.model_properties.dataset << "\n";
                    outfile.close();
                }
                t1 = high_resolution_clock::now();
                // << ", Predictions: " << round(predictions,2)
            }
        }
    }
}

