#include <complex>
#include <torch/torch.h>
#include <iostream>

#include <omp.h>
#include <mpi.h>
#include <CombBLAS/CombBLAS.h>

extern "C" {
	void zdotc(std::complex<double>* retval,
			const int *n,
			const std::complex<double> *zx,
			const int *incx,
			const std::complex<double> *zy,
			const int *incy
			);
}
#define N 5

using namespace std;
using namespace combblas;

#define ITERATIONS 10
#define EDGEFACTOR 8

int main(int argc, char* argv[])
{
	// MPI_Init(NULL, NULL);

	// int nprocs, myrank;
	// MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	// MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	// std::cout << "Hello from process " << myrank << " of " << nprocs << std::endl;
	// AWeighted = new SpParMat<int64_t, double, SpDCCols<int64_t, double>>(MPI_COMM_WORLD);
	 
	// CombBLAS::SpParMat<int, double, CombBLAS::SpDCCols<int,double> > A;



	int nprocs, myrank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	if(argc < 2)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./IndexingTiming <Scale>" << endl;
		}
		MPI_Finalize(); 
		return -1;
	}				

		cout << "i have rank " << myrank << endl;
		typedef SpParMat <int, double, SpDCCols<int,double> > PARDBMAT;
		PARDBMAT *A, *B;		// declare objects
 		double initiator[4] = {.6, .4/3, .4/3, .4/3};
		DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();

		int scale = static_cast<unsigned>(atoi(argv[1]));
		ostringstream outs, outs2, outs3;
		outs << "Forcing scale to : " << scale << endl;
		SpParHelper::Print(outs.str());
		DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, true );        // generate packed edges
		SpParHelper::Print("Generated renamed edge lists\n");
		
		// conversion from distributed edge list, keeps self-loops, sums duplicates
		A = new PARDBMAT(*DEL, false); 	// already creates renumbered vertices (hence balanced)
		delete DEL;	// free memory before symmetricizing
		SpParHelper::Print("Created double Sparse Matrix\n");

		A->PrintInfo();
		// auto tmp = A->spSeq;
		// cout << tmp << endl;

		// PARDBMAT::Iterator it = A.begnz();

		// while (it != A.endnz()) {
    // int row_index = it.row();
    // int col_index = it.col();
    // double value = *it;
	


	// int n, inca = 1, incb = 1, i;
	// std::complex<double> a[N], b[N], c;
	// n = N;
	
	// for( i = 0; i < n; i++ ){
	// 	a[i] = std::complex<double>(i,i*2.0);
	// 	b[i] = std::complex<double>(n-i,i*2.0);
	// }
	// zdotc(&c, &n, a, &inca, b, &incb );
	// std::cout << "The complex dot product is: " << c << std::endl;

  // 	torch::Tensor tensor = torch::rand({2, 3});
 	// std::cout << tensor.item<float>() << std::endl;
	
	return 0;
}
