#include <complex>
#include <torch/torch.h>
#include <iostream>

#include <omp.h>
#include <mpi.h>
#include <CombBLAS/CombBLAS.h>
#include <CombBLAS/SpParMat.h>

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

typedef int IT;
typedef double NT;
typedef SpDCCols<IT, NT> DER;
typedef PlusTimesSRing<NT, NT> SR;




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

	// if(argc < 4){
	// 		if(myrank == 0)
	// 		{
	// 				cout << "Usage: ./<Binary> <MatrixA> <MatrixB> <MatrixCC>" << endl;
	// 		}
	// 		MPI_Finalize();
	// 		return -1;
	// }		

	// string Aname(argv[1]);
	// string Bname(argv[2]);
	// string CCname(argv[3]);

	shared_ptr<CommGrid> fullWorld;
	fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
	
	// SpParMat<int64_t, double, SpDCCols < int64_t, double >> A2D(fullWorld);
	// SpParMat<int64_t, double, SpDCCols < int64_t, double >> B2D(fullWorld);
	// SpParMat<int64_t, double, SpDCCols < int64_t, double >> CC2D(fullWorld);

	// A2D.ParallelReadMM(Aname, true, maximum<double>());
	// B2D.ParallelReadMM(Bname, true, maximum<double>());
	// CC2D.ParallelReadMM(CCname, true, maximum<double>());
	
	cout << "i have rank " << myrank << endl;
	typedef SpParMat <IT, NT, DER > PARDBMAT;
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

	int count = 0;	
	int total = 0;

	cout << endl;
	cout << "i have rank " << myrank << " and: " << A->getSpSeq()->getnrow() << endl;
	SpParHelper::Print("ncols");


	vector<double> diag = {3.0, 2.0, 1.0, 1.5};
	FullyDistVec<int, double> *scaler = new FullyDistVec<int, double>(diag, fullWorld);

	SpCCols<int,double> *ARecv = new SpCCols<int,double>(*(A->getSpSeq()));
	Csc<int, double> *Acsc = ARecv->GetCSC();
	cout << "number of cols: " << Acsc->n << endl;

	SpParHelper::Print("ncols" + A->getSpSeq()->getnrow());


	// for(SpDCCols<int,double>::SpColIter colit = A->seq().begcol(); colit != A->seq().endcol(); ++colit)	// iterate over columns
	// {
	// 	for(SpDCCols<int,double>::SpColIter::NzIter nzit = A->seq().begnz(colit); nzit != A->seq().endnz(colit); ++nzit)
	// 	{	

	// 		cout << "before: " << nzit.rowid() << '\t' << colit.colid() << '\t' << nzit.value() << '\n';	
	// 		nzit.scale_value(3.0);
	// 		cout << "after: " << nzit.rowid() << '\t' << colit.colid() << '\t' << nzit.value() << '\n';	
	// 	}
	// }	

	// MPI_Allreduce( &count, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	if (myrank == 0){
		if(total == A->getnnz()){
			SpParHelper::Print( "Iteration passed soft test\n");
			cout << total << endl;
		}
		else
			SpParHelper::Print( "Iteration failed !!!\n") ;
	}

	// // PARDBMAT out;
	// PARDBMAT D = *A;
	// PARDBMAT C;
	// // out = new PARDBMAT(); 
	// // auto tmpe = Mult_AnXBn_Synch(A,out);
	// SpParMat<int64_t, double, SpDCCols < int64_t, double >> out;

	// out = PSpGEMM<SR, int64_t, double, double, SpDCCols < int64_t, double >, SpDCCols <int64_t, double>>(A2D,B2D);		

	// cout << "hoi" << endl;

	// if(CC2D == out){
	// 		if(myrank == 0) fprintf(stderr, "Correct\n");
	// }
	// else{
	// 		if(myrank == 0) fprintf(stderr, "Not correct\n");
	// }

	// count = 0;
	// total = 0;
	
	// for(SpDCCols<int,double>::SpColIter colit = out.seq().begcol(); colit != out.seq().endcol(); ++colit)	// iterate over columns
	// {
	// 	for(SpDCCols<int,double>::SpColIter::NzIter nzit = out.seq().begnz(colit); nzit != out.seq().endnz(colit); ++nzit)
	// 	{	
	// 		cout << "from rank " << myrank << ": " << nzit.rowid() << '\t' << colit.colid() << '\t' << nzit.value() << '\n';	
	// 		count++;
	// 	}
	// }	

	// MPI_Allreduce( &count, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	// SpParHelper::Print("finished");
	// // int n, inca = 1, incb = 1, i;
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
	
	MPI_Finalize();
	
	return 0;
}
