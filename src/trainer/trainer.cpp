#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <time.h>
#include "trainer.h"

#include "../model/model.h"
#include "../utils/scores.h"
#include "../utils/configParse.h"

#define BILLION 1000000000L
#define MILLION 1000000L

using LossFunction = at::Tensor(*)(const at::Tensor&, const at::Tensor&);


void train_model(const ConfigProperties& config, torch::Tensor &labels, torch::Tensor &input_features, LossFunction loss_fn, BaseModel *model){
    
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
    
    std::ofstream myfile;
    myfile.open ("training-stats.csv");
    myfile << "epoch,epoch_time,total_time,train_loss,test_loss,test_acc,test_f1\n";
    myfile.close();
    
    struct timespec start, end, start_epoch, end_epoch;
    uint64_t diff, diff_to_start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    clock_gettime(CLOCK_MONOTONIC_RAW, &start_epoch);
    
    for (int epoch = 0; epoch < n_epochs; epoch++){
        torch::Tensor predictions = model->forward(input_features);
        torch::Tensor train_predictions = predictions.index({at::indexing::Slice(0,train_set_cutoff)});
        torch::Tensor loss = loss_fn(train_predictions, train_labels);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        lr_scheduler.step();

        if (epoch % stepsize_output == 0){
            {
                torch::NoGradGuard no_grad;
                torch::Tensor test_predictions = predictions.index({at::indexing::Slice(train_set_cutoff,labels.size(0))});
                torch::Tensor test_loss = loss_fn(test_predictions, test_labels);
                torch::Tensor acc = accuracy(test_predictions, test_labels);
                torch::Tensor f1 = f1_score(test_predictions, test_labels, 40);

                std::cout << "Epoch [" << epoch << "/" << n_epochs << "], ";
                std::cout << "Train Loss: " << loss.item<double>() << ", ";
                std::cout << "Test Loss: " << test_loss.item<double>() << ", ";
                std::cout << "Test Accuracy: " << acc.item<double>() << ", ";
                std::cout << "Test F1: " << f1.item<double>() << std::endl;


                clock_gettime(CLOCK_MONOTONIC_RAW, &end_epoch);
                diff = (BILLION * (end_epoch.tv_sec - start_epoch.tv_sec) + end_epoch.tv_nsec - start_epoch.tv_nsec) / BILLION;
	            printf("epoch time = %llu seconds\n", (long long unsigned int) diff);
                diff_to_start = (BILLION * (end_epoch.tv_sec - start.tv_sec) + end_epoch.tv_nsec - start.tv_nsec) / BILLION;
                printf("total time = %llu seconds\n", (long long unsigned int) diff_to_start);
                std::ofstream myfile;
                myfile.open("training-stats.csv", std::ios_base::app);
                myfile << epoch << "," << diff << "," << diff_to_start << "," << loss.item<double>() << "," << test_loss.item<double>() << "," << acc.item<double>() << "," << f1.item<double>() << "\n";
                myfile.close();
                clock_gettime(CLOCK_MONOTONIC_RAW, &start_epoch);
                // << ", Predictions: " << round(predictions,2)
            }
        }
    }
}

