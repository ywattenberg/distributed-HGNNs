#include <iostream>
#include <torch/torch.h>

class matrix
{
private:
    torch::Tensor H; /* Sparse Matrix */
    torch::Tensor X;       /* Vertex Feature */
    torch::Tensor De, Dv;  /* Diagonals extracted from `H` and `X` */

public:
    matrix(torch::Tensor H, torch::Tensor X);
    ~matrix();
    torch::Tensor getH();
    torch::Tensor getX();
    torch::Tensor getDe();
    torch::Tensor getDv();
    void setH(torch::Tensor H);
    void setX(torch::Tensor X);
    void calculateDe();
    void calculateDv();
};

