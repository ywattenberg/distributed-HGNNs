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

int main()
{
	MPI_Init(NULL, NULL);

	int nprocs, myrank;
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	std::cout << "Hello from process " << myrank << " of " << nprocs << std::endl;
	 AWeighted = new SpParMat<int64_t, double, SpDCCols<int64_t, double>>(MPI_COMM_WORLD);
	 
	CombBLAS::SpParMat<int, double, SpDCCols<int,double> > A;
	return 0;
}
