#include <complex>
#include <torch/torch.h>
#include <iostream>

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
	int n, inca = 1, incb = 1, i;
	std::complex<double> a[N], b[N], c;
	n = N;
	
	for( i = 0; i < n; i++ ){
		a[i] = std::complex<double>(i,i*2.0);
		b[i] = std::complex<double>(n-i,i*2.0);
	}
	zdotc(&c, &n, a, &inca, b, &incb );
	std::cout << "The complex dot product is: " << c << std::endl;

  	torch::Tensor tensor = torch::rand({2, 3});
 	std::cout << tensor.item<float>() << std::endl;
	
	return 0;
}
