#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/SpParMat3D.h"
#include "CombBLAS/ParFriends.h"
#include "CombBLAS/FullyDistVec.h"
#include "CombBLAS/SpParMat.h"
#include "utils/DenseMatrix.h"
#include "utils/LossFn.h"

using namespace std;
using namespace combblas;

#define EPS 0.0001

#ifdef _OPENMP
int cblas_splits = omp_get_max_threads();
#else
int cblas_splits = 1;
#endif


// Simple helper class for declarations: Just the numerical type is templated
// The index type and the sequential matrix type stays the same for the whole code
// In this case, they are "int" and "SpDCCols"
template <class NT>
class PSpMat
{
public:
    typedef SpDCCols < int64_t, NT > DCCols;
    typedef SpParMat < int64_t, NT, DCCols > MPI_DCCols;
    typedef FullyDistVec < int64_t, NT > MPI_DenseVector;
};

int main(int argc, char* argv[])
{
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    if(argc < 4){
        if(myrank == 0)
        {
            cout << "Usage: ./<Binary> <MatrixA> <MatrixB> <MatrixCC>" << endl;
        }
        MPI_Finalize();
        return -1;
    }
    else {

        string Aname(argv[1]);
        string Bname(argv[2]);
        string CCname(argv[3]);
        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
        
        double t0, t1;
        
        typedef PlusTimesSRing<double, double> PTFF;

        SpParMat<int64_t, double, SpDCCols < int64_t, double >> A2D(fullWorld);
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> B2D(fullWorld);
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> CC2D(fullWorld);

        A2D.ParallelReadMM(Aname, true, maximum<double>());
        B2D.ParallelReadMM(Bname, true, maximum<double>());
        CC2D.ParallelReadMM(CCname, true, maximum<double>());


        SpTuples<int64_t, double>* tuples = new SpTuples<int64_t, double>(0, 4, 4);
        std::cout << "rank " << myrank << " has no work" << std::endl;
        SpDCCols<int64_t, double> * out =  new SpDCCols<int64_t, double>(*tuples, false);
        std::cout << "rank " << myrank << " has  still no work" << std::endl;
        SpParMat<int64_t, double, SpDCCols<int64_t, double>> blabla = SpParMat<int64_t, double, SpDCCols<int64_t, double>> (out, fullWorld);

        // int64_t length = 50;
        // FullyDistVec<int64_t, double> v = FullyDistVec(length, 2);
        // auto tmp = v.get_values();

        // auto out =  PSpGEMM<PTFF, int64_t, double, double, SpDCCols < int64_t, double >, SpDCCols <int64_t, double >>(A2D,B2D);



        std::vector<double> tmp = {4.0, 1.0, 3.0, 2.0, 5.0 ,1.0, 4.0, 1.0, 0.0};

        if (myrank == 0){
            cout << "length of vector: " << tmp.size() << endl;
            for (int i = 0; i < tmp.size(); i++){
                cout << "value: " << tmp.at(i) << endl;
            }
        }

        A2D.PrintInfo();
        MPI_Barrier(MPI_COMM_WORLD);

        SpParMat<int64_t, double, SpDCCols<int64_t, double>> res = PSpSCALE<PTFF, int64_t, double, SpDCCols<int64_t, double>>(A2D, tmp);

        res.ParallelWriteMM("../data/m_g_ms_gs/bla.mtx", true); 

        const std::string filename = "../data/m_g_ms_gs/dense-test.mtx";
        DenseMatrix<double> simi(4, 5, fullWorld);

        simi.ParallelReadDMM(filename, false);
        simi.printLocalMatrix();
        // if (res == CC2D){
        //     if (myrank == 0){
        //         fprintf(stderr, "Correct\n");
        //     }
        // } else{
        //     if(myrank == 0) fprintf(stderr, "Not correct\n");
        // }

        if(myrank == 1) fprintf(stderr, "***\n");

        // auto out =  PSpGEMM<PTFF, int64_t, double, double, SpDCCols < int64_t, double >, SpDCCols <int64_t, double >>(A2D,B2D);

        // int N = 4;
        // int M = 4;
        // double *matrix_data = NULL;
        // if (myrank == 0) {
        //     matrix_data = (double *)malloc(N * N * sizeof(double));
        //     // Initialize matrix_data with some values
        //     for (int i = 0; i < N * N; i++) {
        //         matrix_data[i] = i + 1; // Just an example initialization
        //     }
        // }

        // // Allocate memory for the local block
        // int local_size = N / M;
        // double *local_block = (double *)malloc(local_size * N * sizeof(double));

        // // Scatter the matrix data
        // MPI_Scatter(matrix_data, local_size * N, MPI_INT, local_block, local_size * N, MPI_INT, 0, MPI_COMM_WORLD);

        // std::vector<double> test;
        // // Scatter the matrix data
        //     if (myrank == 0){
        //         test = { 16,0.6,6,7,
        //                  1,3.6,0,4.1,
        //                  1,5,9.5,0,
        //                  13, 6, 0.2, 3.5};

        //     }

        std::vector<double> test = { 16,0.6,6,7,
                1,3.6,0,4.1,
                1,5,9.5,0,
                13, 6, 0.2, 3.5};
        

        std::vector<double>* dist;
        double* d_array;

        switch(myrank){
            case 0: 
                d_array = new double[]{16,0.6,1,3.6};
                dist = new std::vector<double>(d_array, d_array + 4);
                break;
            case 1:
                d_array = new double[]{6,7,0,4.1};
                dist = new std::vector<double>(d_array, d_array + 4);
                break;
            case 2:
                d_array = new double[]{1,5,13,6};
                dist = new std::vector<double>(d_array, d_array +4);
                break;
            case 3:
                d_array = new double[]{9.5,0,0.2,3.5};
                dist = new std::vector<double>(d_array, d_array + 4);
                break;
        }

        std::vector<double>* dist2;
        
        switch(myrank){
            case 0: 
                d_array = new double[]{16,0.6,1,3.6};
                dist2 = new std::vector<double>(d_array, d_array + 4);
                break;
            case 1:
                d_array = new double[]{6,7,0,4.1};
                dist2 = new std::vector<double>(d_array, d_array + 4);
                break;
            case 2:
                d_array = new double[]{1,5,13,6};
                dist2 = new std::vector<double>(d_array, d_array + 4);
                break;
            case 3:
                d_array = new double[]{9.5,0,0.2,3.5};
                dist2 = new std::vector<double>(d_array, d_array + 4);
                break;
        }


        // for (int i = 0; i < 4; i++){
        //     dist[i] = local_block[i];
        // }

        // cout << "hello from rank " << myrank << endl;
        int rows = 4;
        int cols = 4;


        DenseMatrix<double> denseTest = DenseMatrix<double>(2,2,dist, fullWorld);
        DenseMatrix<double> denseTest2 = DenseMatrix<double>(2,2,dist2, fullWorld);

        CrossEntropyLoss<PTFF, double>(denseTest, dist, true);

        // if (myrank == 2){
        //     cout << "from rank " << myrank << " ";
        //     vector<double> forPrint = *denseTest.getValues();
        //     for (int i = 0; i < 4; i++){
        //         cout << forPrint[i] << " ";
        //     }

        // }

        DenseMatrix<double> output = DenseDenseMult<PTFF, double>(denseTest, denseTest2);
        
        
        // DenseMatrix<double> output = SpDenseMult<PTFF, int64_t, double, SpDCCols < int64_t, double >>(res, denseTest);
        std::vector<double> outValuesLocal = *output.getValues();


        if (myrank == 0){
            for (int i = 0; i < output.getLocalRows(); i++){
                for (int j = 0; j < output.getLocalCols(); j++){
                    cout << outValuesLocal[i*output.getLocalCols() + j] << " ";
                }
                cout << endl;
            }
        }
        
    //     // Increase number of layers 1 -> 4 -> 16
    //     // for(int layers = 1; layers <= 16; layers = layers * 4){
            
    //     //     if(myrank == 0) fprintf(stderr, "Trying %d layers\n", layers);

            

    //     //     // Convert 2D matrices to 3D
    //     //     SpParMat3D<int64_t, double, SpDCCols < int64_t, double >> A3D(A2D, layers, true, false);
    //     //     SpParMat3D<int64_t, double, SpDCCols < int64_t, double >> B3D(B2D, layers, false, false);

    //     //     SpParMat3D<int64_t, double, SpDCCols < int64_t, double >> C3D = 
    //     //         Mult_AnXBn_SUMMA3D<PTFF, double, SpDCCols<int64_t, double>, int64_t, double, double, SpDCCols<int64_t, double>, SpDCCols<int64_t, double> >
    //     //         (A3D, B3D);
    //     //     SpParMat<int64_t, double, SpDCCols < int64_t, double >> C3D2D = C3D.Convert2D();

    //     //     if(CC2D == C3D2D){
    //     //         if(myrank == 0) fprintf(stderr, "Correct\n");
    //     //     }
    //     //     else{
    //     //         if(myrank == 0) fprintf(stderr, "Not correct\n");
    //     //     }
            
    //     //     if(myrank == 0) fprintf(stderr, "***\n");
    //     // }

    //     // double arr[4][2] = {1234.0, 56.0, 1212.0, 33.0, 1434.0, 80.0, 1312.0, 78.0};

    //     // DenseParMat<int64_t, double> test(3.0, fullWorld, 4,4);

    //     // double **arr_test = test.getArray();


    //     // for(SpDCCols<int,double>::SpColIter colit = A2D.seq().begcol(); colit != A2D.seq().endcol(); ++colit)	// iterate over columns
	//     // {
    //     //     for(SpDCCols<int,double>::SpColIter::NzIter nzit = A2D.seq().begnz(colit); nzit != A2D.seq().endnz(colit); ++nzit)
    //     //     {	
    //     //         cout << "before: " << nzit.rowid() << '\t' << colit.colid() << '\t' << nzit.value() << '\n';	
    //     //         nzit.scale_value(3.0);
    //     //         cout << "after: " << nzit.rowid() << '\t' << colit.colid() << '\t' << nzit.value() << '\n';	

    //     //     }
	//     // }

        
    }
    MPI_Finalize();
    return 0;
}