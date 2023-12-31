#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include "trainer.h"
#include "../model/model.h"
#include "../utils/scores.h"
#include "../utils/configParse.h"
#include "../utils/LossFn.h"
#include "../model/dist-model.h"
#include "../DenseMatrix/DenseMatrix.h"

#define BILLION 1000000000L
#define MILLION 1000000L

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

using namespace std;
using namespace combblas;

typedef combblas::PlusTimesSRing<double, double> PTFF;

void train_dist_model(const ConfigProperties& config, std::vector<int> &labels, DenseMatrix<double> &input, DistModel *model, int run_id, bool timing, std::string timing_file) {

    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    double lr = config.trainer_properties.learning_rate;
    int n_epochs = config.trainer_properties.epochs;
    int stepsize_output = config.trainer_properties.output_stepsize;
    long train_set_cutoff = config.data_properties.test_idx;

    int lr_step_size = config.lr_scheduler_properties.step_size;
    double lr_gamma = config.lr_scheduler_properties.gamma;

    auto t1 = high_resolution_clock::now();
    for (int epoch = 0; epoch < n_epochs; epoch++){

      DenseMatrix<double> res = model->forward(input);

      double loss = CrossEntropyLoss<PTFF, double>(res, &labels, train_set_cutoff);
      model->backward(res, &labels, lr);

      if(epoch % lr_step_size == 0){
        lr = lr * lr_gamma;
      }


      if (epoch % stepsize_output == 0){

        auto t2 = high_resolution_clock::now();
        /* Getting number of milliseconds as an integer. */
        auto ms_int = duration_cast<milliseconds>(t2 - t1);
        /* Getting number of milliseconds as a double. */
        duration<double, std::milli> ms_double = t2 - t1;

        std::vector<double> loss_vec = LossMetrics<PTFF, double>(res, &labels, train_set_cutoff);
        double train_loss = loss_vec[0];
        double test_loss = loss_vec[1];
        double test_acc = loss_vec[2];
        double test_f1 = loss_vec[3];

        if (myrank == 0) {
          std::cout << "Epoch [" << epoch << "/" << n_epochs << "], " << ms_int.count() << "ms, ";
          std::cout << "Train Loss: " << train_loss << ", ";
          std::cout << "Test Loss: " << test_loss << ", ";
          std::cout << "Test Accuracy: " << test_acc << ", ";
          std::cout << "Test F1: " << test_f1 << std::endl;
        }


        if (timing && myrank == 0){
            std::ofstream outfile;
            outfile.open(timing_file, std::ios_base::app);
            // append time to csv file
            outfile << run_id << "," << epoch << "," << ms_int.count() << "," << train_loss << "," << test_loss << "," << test_acc << "," << test_f1 << "\n";
            outfile.close();
        }
        t1 = high_resolution_clock::now();
      }
    }
}