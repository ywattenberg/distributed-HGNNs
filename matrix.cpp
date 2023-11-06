#include "matrix.h"



matrix::matrix(torch::Tensor H, torch::Tensor X)
{
    this->H = H;
    this->X = X;
    calculateDe();
    calculateDv();
    return;
}


matrix::~matrix()
{
    
}

torch::Tensor matrix::getH()
{
    return H;
}

torch::Tensor matrix::getX()
{
    return X;
}

torch::Tensor matrix::getDe()
{
    return De;
}

torch::Tensor matrix::getDv()
{
    return Dv;
}

void matrix::setH(torch::Tensor H)
{
    // Check if given H is sparse i.e. of layout is sparse
    if (H.layout() != torch::kSparse)
    {
        std::cout << "Given H is not sparse. Converting given Tensor" << std::endl;
        H = H.to_sparse();
    }

    this->H = H;
}

void matrix::setX(torch::Tensor X)
{
    this->X = X;
}

void matrix::calculateDe()
{
    
}

void matrix::calculateDv()
{
    
}

