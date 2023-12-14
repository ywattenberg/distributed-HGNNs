using namespace std;

double* PDGEMM(const double A[], const double B[], int ROWS_A, int COLS_B, int COLS_A);

std::vector<double>* VPDGEMM(std::vector<double>* A, std::vector<double>* B, int ROWS_A, int COLS_A, int ROWS_B, int COLS_B);