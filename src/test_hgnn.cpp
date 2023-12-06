#include "test_hgnn.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <mpi.h>

// others might be needed
// #include "hgnn.hpp"          //TODO: finish hgnn class then add this
#include "distributed_sparse.h"
#include "15D_dense_shift.hpp"
#include "15D_sparse_shift.hpp"
#include "25D_cannon_dense.hpp"
#include "25D_cannon_sparse.hpp"

#include "SpmatLocal.hpp"
#include "FlexibleGrid.hpp"

#include "common.h"
#include "sparse_kernels.h"

using namespace std;


void benchmark_algorithm(SpmatLocal* spmat, 
        string algorithm_name,
        string output_file,
        bool fused,
        int R,
        int c,
        string app 
) {
    // initialize MPI and distributed matrix kernels
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ofstream fout;
        fout.open(output_file, std::ios_base::app 
    );

    StandardKernel local_ops;
    Distributed_Sparse* d_ops;

    // TODO: initialize algorithm 



    // run algorithm and record time information
    if(rank == 0) {
        cout << "Starting benchmark " << app << endl;
    }

    d_ops->reset_performance_timers();
    my_timer_t t = start_clock();
    int num_trials = 0;

    double application_communication_time = 0.0;

    do {
        num_trials++;

        if(app == "vanilla") {  // test vanilla spmm
            if(fused) {
                d_ops->fusedSpMM(A, 
                        B, 
                        S, 
                        sddmm_result, 
                        Amat);
            }
            else {
                d_ops->sddmmA(A, B, S, sddmm_result);
                d_ops->spmmA(A, B, S);
            } 
        }
        // else if(app=="hgnn") { //TODO: finish model class and test hgnn
        //     gnn->forward();
        // }
    } while(num_trials < 5);
    MPI_Barrier(MPI_COMM_WORLD);

    // finish timing and write output file
    json j_obj; 

    double elapsed = stop_clock_get_elapsed(t);
    double ops = 2 * spmat->dist_nnz * 2 * R * num_trials;
    double throughput = ops / elapsed;
    throughput /= 1e9;

    j_obj["elapsed"] = elapsed;
    j_obj["overall_throughput"] = throughput;
    j_obj["fused"] = fused;
    j_obj["num_trials"] = num_trials;
    j_obj["alg_name"] = algorithm_name;
    j_obj["alg_info"] = d_ops->json_algorithm_info();
    j_obj["application_communication_time"] = application_communication_time; 
    j_obj["perf_stats"] = d_ops->json_perf_statistics();

    if(rank == 0) {
        fout << j_obj.dump(4) << "," << endl;
    } 

    fout.close();

    delete d_ops;
}        